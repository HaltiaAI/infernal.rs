use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::{
    kv::{Kv, Type, Value},
    pad, Dims, LoadError, TensorType,
};

pub struct Context {
    pub header: Header,

    pub kv: Vec<Kv>,
    pub tensors: Vec<TensorInfo>,

    pub alignment: usize,
    /// The offset of the data section in the file.
    pub data_offset: usize,
    /// The total size of the data section only.
    pub data_size: usize,
}

#[derive(Debug)]
pub struct Header {
    pub magic: u32,
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: Vec<u8>,
    // Number of elements, per dimension.
    pub dims: Dims,
    pub typ: TensorType,
    // Offset relative to `Context::data_offset`. Must be a multiple of [`WEIGHT_ALIGN`].
    pub offset: usize,
}

pub const WEIGHT_ALIGN: usize = 32;
const MEM_ALIGN: usize = 16;
const GGUF_MAGIC: u32 = 1179993927;

impl TensorInfo {
    pub fn stride(&self) -> [usize; Dims::MAX] {
        let mut stride = [0; 4];
        stride[0] = self.typ.size();
        stride[1] = stride[0] * (self.dims[0] / self.typ.block_size());
        for i in 2..Dims::MAX {
            stride[i] = stride[i - 1] * self.dims[i - 1]
        }
        stride
    }

    pub fn padded_size(&self) -> usize {
        pad(self.size(), MEM_ALIGN)
    }

    pub fn size(&self) -> usize {
        let stride = self.stride();
        let mut size = self.dims[0] * stride[0] / self.typ.block_size();
        #[allow(clippy::needless_range_loop)]
        for i in 1..self.dims.n() {
            size += (self.dims[i] - 1) * stride[i];
        }
        size
    }
}

impl Context {
    pub fn load<R: Read + Seek>(reader: R) -> Result<(Self, R), LoadError> {
        let reader = BufReader::new(reader);
        let mut loader = Loader { reader };

        let header = loader.read_header()?;
        let kv = loader.read_kv_pairs(&header)?;

        let mut alignment = WEIGHT_ALIGN;
        for kv in &kv {
            if kv.key == b"general.alignment" {
                if let Value::Uint32(a) = kv.value {
                    alignment = a as usize;
                    break;
                }
            }
        }

        let (size, tensors) = loader.read_tensor_infos(&header, alignment)?;

        // Calc the current file offset - this is where the data section starts.
        let mut offset = loader.reader.stream_position()? as usize;
        {
            let alignment = alignment;
            let offset_pad = offset % alignment;
            if offset_pad != 0 {
                offset += alignment - offset_pad;
                loader.reader.seek(SeekFrom::Start(offset as u64))?;
            }
        }

        Ok((
            Self {
                header,
                kv,
                tensors,
                alignment,
                data_offset: offset,
                data_size: size,
            },
            loader.reader.into_inner(),
        ))
    }

    pub fn get(&self, key: &[u8]) -> Option<&Value> {
        // TODO#perf: use a hashmap here? usually this vec is very small so may not be necessary
        self.kv.iter().find(|kv| kv.key == key).map(|kv| &kv.value)
    }
}

struct Loader<R: BufRead> {
    reader: R,
}

impl<R: BufRead> Loader<R> {
    fn read_header(&mut self) -> Result<Header, LoadError> {
        let magic = self.reader.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(LoadError::UnsupportedFileVersion(magic));
        };

        let version = self.reader.read_u32::<LittleEndian>()?;

        let n_tensors = self.version_dependent_read_u64(version)?;
        let n_kv = self.version_dependent_read_u64(version)?;

        Ok(Header {
            magic,
            version,
            n_tensors,
            n_kv,
        })
    }

    fn read_kv_pairs(&mut self, header: &Header) -> Result<Vec<Kv>, LoadError> {
        let mut kv = Vec::with_capacity(header.n_kv as usize);
        for _ in 0..header.n_kv {
            let key = self.read_gguf_string(header.version)?;
            let value = self.read_gguf_value(header.version, None)?;
            kv.push(Kv { key, value });
        }
        Ok(kv)
    }

    fn read_gguf_value(
        &mut self,
        version: u32,
        in_array: Option<Type>,
    ) -> Result<Value, LoadError> {
        let typ = in_array.map(Ok).unwrap_or_else(|| self.read_gguf_type())?;
        // TODO#perf: read all values in one go if this is an array.
        Ok(match typ {
            Type::Uint8 => Value::Uint8(self.reader.read_u8()?),
            Type::Int8 => Value::Int8(self.reader.read_i8()?),
            Type::Uint16 => Value::Uint16(self.reader.read_u16::<LittleEndian>()?),
            Type::Int16 => Value::Int16(self.reader.read_i16::<LittleEndian>()?),
            Type::Uint32 => Value::Uint32(self.reader.read_u32::<LittleEndian>()?),
            Type::Int32 => Value::Int32(self.reader.read_i32::<LittleEndian>()?),
            Type::Float32 => Value::Float32(self.reader.read_f32::<LittleEndian>()?),
            Type::Bool => Value::Bool(self.reader.read_u8()? != 0),
            Type::String => Value::String(self.read_gguf_string(version)?),
            Type::Array => {
                if in_array.is_some() {
                    return Err(LoadError::NestedArray);
                } else {
                    Value::Array(self.read_gguf_array(version)?)
                }
            }
            Type::Uint64 => Value::Uint64(self.reader.read_u64::<LittleEndian>()?),
            Type::Int64 => Value::Int64(self.reader.read_i64::<LittleEndian>()?),
            Type::Float64 => Value::Float64(self.reader.read_f64::<LittleEndian>()?),
        })
    }

    fn read_tensor_type(&mut self) -> Result<TensorType, LoadError> {
        // TODO: this is i32 in the original impl but this value can never be negative, so..?
        let typ = self.reader.read_u32::<LittleEndian>()?;
        typ.try_into()
    }

    fn read_gguf_array(&mut self, version: u32) -> Result<Vec<Value>, LoadError> {
        let typ = self.read_gguf_type()?;
        let len = self.version_dependent_read_u64(version)?;
        let mut array = Vec::with_capacity(len as usize);
        // TODO#perf: optimize this to a single write like in ggml
        for _ in 0..len {
            let value = self.read_gguf_value(version, Some(typ))?;
            array.push(value);
        }
        Ok(array)
    }

    /// Reads info for all tensors and computes the total size of the data section, taking into account
    /// the alignment.
    fn read_tensor_infos(
        &mut self,
        header: &Header,
        alignment: usize,
    ) -> Result<(usize, Vec<TensorInfo>), LoadError> {
        let mut size = 0;
        let mut infos = Vec::with_capacity(header.n_tensors as usize);
        for _ in 0..header.n_tensors {
            let name = self.read_gguf_string(header.version)?;
            let n_dims = self.reader.read_u32::<LittleEndian>()?;
            let mut dims = Dims::new_blank(n_dims as usize);
            for d in 0..n_dims {
                dims.set_dim(
                    d as usize,
                    self.version_dependent_read_u64(header.version)? as usize,
                );
            }
            let typ = self.read_tensor_type()?;
            let offset = self.reader.read_u64::<LittleEndian>()? as usize;

            let n_elem = dims.total_elems();
            let block_size = typ.block_size();
            if n_elem % block_size != 0 {
                return Err(LoadError::UnalignedTensorElements {
                    name: String::from_utf8_lossy(&name).into_owned(),
                    n_elem,
                    block_size,
                });
            }

            let type_size = typ.size();
            let cur_size = (n_elem * type_size) / block_size;
            size += pad(cur_size, alignment);

            infos.push(TensorInfo {
                name,
                dims,
                typ,
                offset,
            })
        }
        Ok((size, infos))
    }

    fn read_gguf_type(&mut self) -> Result<Type, LoadError> {
        self.reader
            .read_u32::<LittleEndian>()?
            .try_into()
            .map_err(From::from)
    }

    fn read_gguf_string(&mut self, version: u32) -> Result<Vec<u8>, LoadError> {
        let len = self.version_dependent_read_u64(version)?;
        let mut buf = vec![0; len as usize];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Depending on whether it's the first GGUF version or later versions, the value read in
    /// is a 32-bit unsigned integer or a 64-bit, respectively.
    fn version_dependent_read_u64(&mut self, version: u32) -> Result<u64, LoadError> {
        Ok(if version == 1 {
            self.reader.read_u32::<LittleEndian>()? as u64
        } else {
            self.reader.read_u64::<LittleEndian>()?
        })
    }
}
