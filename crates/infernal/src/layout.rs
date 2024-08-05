use core::fmt;

use thiserror::Error;

pub use gguf::Dims;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Type {
    F32,
    F16,
    I32,

    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
}

impl Type {
    #[inline(always)]
    pub fn size(&self) -> usize {
        // NB: For now we just forward to the gguf implementation, but we will likely support other,
        // non-gguf quantization types in the future.
        gguf::TensorType::from(*self).size()
    }

    #[inline(always)]
    pub fn block_size(&self) -> usize {
        gguf::TensorType::from(*self).block_size()
    }

    #[inline(always)]
    pub fn name(&self) -> &'static str {
        gguf::TensorType::from(*self).name()
    }

    #[inline(always)]
    pub fn is_quantized(&self) -> bool {
        gguf::TensorType::from(*self).is_quantized()
    }
}

#[derive(Clone, Copy, Debug, Error)]
#[error("GGUF tensor type '{0}' is not supported")]
pub struct UnsupportedGgufTensorType(pub gguf::TensorType);

impl TryFrom<gguf::TensorType> for Type {
    type Error = UnsupportedGgufTensorType;

    #[inline(always)]
    fn try_from(t: gguf::TensorType) -> Result<Self, Self::Error> {
        Ok(match t {
            gguf::TensorType::F32 => Self::F32,
            gguf::TensorType::F16 => Self::F16,
            gguf::TensorType::I32 => Self::I32,
            gguf::TensorType::Q2_K => Self::Q2_K,
            gguf::TensorType::Q3_K => Self::Q3_K,
            gguf::TensorType::Q4_K => Self::Q4_K,
            gguf::TensorType::Q5_K => Self::Q5_K,
            gguf::TensorType::Q6_K => Self::Q6_K,
            gguf::TensorType::Q8_K => Self::Q8_K,
            _ => return Err(UnsupportedGgufTensorType(t)),
        })
    }
}

impl From<Type> for gguf::TensorType {
    #[inline(always)]
    fn from(t: Type) -> Self {
        match t {
            Type::F32 => Self::F32,
            Type::F16 => Self::F16,
            Type::I32 => Self::I32,
            Type::Q2_K => Self::Q2_K,
            Type::Q3_K => Self::Q3_K,
            Type::Q4_K => Self::Q4_K,
            Type::Q5_K => Self::Q5_K,
            Type::Q6_K => Self::Q6_K,
            Type::Q8_K => Self::Q8_K,
        }
    }
}

/// Supported dimensions.
pub const MAX_DIMS: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D1 {
    pub cols: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D2 {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D3 {
    pub depth: usize,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D4 {
    pub batches: usize,
    pub depth: usize,
    pub rows: usize,
    pub cols: usize,
}

impl From<usize> for D1 {
    fn from(cols: usize) -> Self {
        Self { cols }
    }
}

impl From<D1> for Dims {
    #[inline(always)]
    fn from(d: D1) -> Self {
        Self::from([d.cols].as_slice())
    }
}

impl From<D2> for Dims {
    #[inline(always)]
    fn from(d: D2) -> Self {
        Self::from([d.cols, d.rows].as_slice())
    }
}

impl From<D3> for Dims {
    #[inline(always)]
    fn from(d: D3) -> Self {
        Self::from([d.cols, d.rows, d.depth].as_slice())
    }
}

impl From<D4> for Dims {
    #[inline(always)]
    fn from(d: D4) -> Self {
        Self::from([d.cols, d.rows, d.depth, d.batches].as_slice())
    }
}

#[macro_export]
macro_rules! d {
    ($rows:expr; x $cols:expr) => {
        $crate::layout::D2 {
            rows: $rows,
            cols: $cols,
        }
    };
    ($depth:expr; x $rows:expr; x $cols:expr) => {
        $crate::layout::D3 {
            depth: $depth,
            rows: $rows,
            cols: $cols,
        }
    };
    ($batches:expr; x $depth:expr; x $rows:expr; x $cols:expr) => {
        $crate::layout::D4 {
            batches: $batches,
            depth: $depth,
            rows: $rows,
            cols: $cols,
        }
    };
}

pub const BATCH: usize = 3;
pub const DEPTH: usize = 2;
pub const ROW: usize = 1;
pub const COL: usize = 0;

/// The number of elements in a tensor, per dimension, up to 4 dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Layout {
    /// Dimensions are stored in reverse order. E.g. for a 2D tensor, ne[0] is column dimension and
    /// ne[1] is the row dimension.
    // TODO: merge Dims into Layout
    pub(crate) dims: [usize; MAX_DIMS], // [col, row, depth, batch]
    /// Stride in bytes.
    ///
    /// For a 2D tensor, stride[0] is the stride per element and stride[1] is the stride for a row, so
    /// e.g. an element at row 1 and col 2 will be 1nb[1] + 2nb[0]. Therefore for the first
    /// 2 dimensions the byte stride is switched up.
    ///
    /// stride[0] = type_size(type)
    /// stride[1] = stride[0]   * (ne[0] / ggml_blck_size(type)) + padding TODO padding ???
    /// stride[i] = stride[i-1] * ne[i-1]
    pub(crate) stride: [usize; MAX_DIMS],
    pub(crate) n: usize,
    typ: Type,
}

impl Layout {
    #[inline(always)]
    pub fn new(dims: impl Into<Dims>, typ: Type) -> Self {
        Self::new_raw_dims(dims.into().as_slice(), typ)
    }

    #[inline(always)]
    // #[deprecated(note = "use `new`")]
    pub fn new_1d(cols: usize, typ: Type) -> Self {
        Self::new_raw_dims(&[cols], typ)
    }

    #[inline(always)]
    // #[deprecated(note = "use `new`")]
    pub fn new_2d(rows: usize, cols: usize, typ: Type) -> Self {
        Self::new_raw_dims(&[cols, rows], typ)
    }

    #[inline(always)]
    // #[deprecated(note = "use `new`")]
    pub fn new_3d(depth: usize, rows: usize, cols: usize, typ: Type) -> Self {
        Self::new_raw_dims(&[cols, rows, depth], typ)
    }

    #[inline(always)]
    // #[deprecated(note = "use `new`")]
    pub fn new_4d(batches: usize, depth: usize, rows: usize, cols: usize, typ: Type) -> Self {
        Self::new_raw_dims(&[cols, rows, depth, batches], typ)
    }

    /// NOTE: The dimensions are stored in reverse order like ggml does, so the first element is the
    /// the column, while the last element is the batch number.
    pub(crate) fn new_raw_dims(dims: &[usize], typ: Type) -> Self {
        let n = dims.len();
        assert!(n <= MAX_DIMS, "maximum number of dimensions is {MAX_DIMS}");

        let mut dims_arr = [1; 4];
        for (i, dim) in dims.iter().enumerate() {
            dims_arr[i] = *dim;
        }
        let dims = dims_arr;

        let mut stride = [0; 4];
        stride[0] = typ.size();
        stride[1] = stride[0] * (dims[0] / typ.block_size());
        for i in 2..MAX_DIMS {
            stride[i] = stride[i - 1] * dims[i - 1];
        }

        Self {
            // While we only use n_dims of this array, we still initialize everything to 1 in case
            // we need to pass it to shaders, which expect a 1 initialized array (1 being the identity
            // in multiplication).
            dims,
            stride,
            n,
            typ,
        }
    }

    #[doc(hidden)]
    pub fn from_raw(
        n: usize,
        dims: [usize; MAX_DIMS],
        stride: [usize; MAX_DIMS],
        typ: Type,
    ) -> Self {
        Self {
            dims,
            stride,
            n,
            typ,
        }
    }

    /// Returns the number of dimension, N.
    ///
    /// To return the elements pair dimensions, use [`Self::dims`].
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.n
    }

    #[doc(hidden)]
    pub fn dims_array(&self) -> [usize; MAX_DIMS] {
        self.dims
    }
    #[doc(hidden)]
    pub fn stride_array(&self) -> [usize; MAX_DIMS] {
        self.stride
    }

    /// Returns a list of elements per dimension.
    ///
    /// NOTE: The list is reversed, so the first element is the last dimension, the column. However,
    /// use [`BATCH`], [`DEPTH`], [`ROW`] and [`COL`] to access the dimensions by name to avoid
    /// confusion.
    #[inline(always)]
    pub fn raw_dims(&self) -> &[usize] {
        &self.dims[..self.n]
    }

    /// The stride specifies where the dimension starts in the associated byte buffer.
    ///
    /// NOTE: The list is reversed, so the first element is the last dimension, the column. However,
    /// use [`BATCH`], [`DEPTH`], [`ROW`] and [`COL`] to access the dimensions by name to avoid
    /// confusion.
    #[inline(always)]
    pub fn raw_stride(&self) -> &[usize] {
        &self.stride[..self.n]
    }

    #[inline(always)]
    pub fn typ(&self) -> Type {
        self.typ
    }

    /// Returns the total number of rows, that is: `batches * depth * rows`.
    #[inline(always)]
    pub fn total_rows(&self) -> usize {
        self.dims[ROW] * self.dims[DEPTH] * self.dims[BATCH]
    }

    /// Returns the total number of elements (columns) in the tensor.
    ///
    /// E.g. if this represents a 2-dimensional tensor, then dim0 * dim1 is going to be returned.
    #[inline(always)]
    pub fn total_elems(&self) -> usize {
        self.raw_dims().iter().product()
    }

    #[inline(always)]
    pub fn batches(&self) -> usize {
        self.dims[BATCH]
    }

    #[inline(always)]
    pub fn batch_stride(&self) -> usize {
        self.stride[BATCH]
    }

    #[inline(always)]
    pub fn depth(&self) -> usize {
        self.dims[DEPTH]
    }

    #[inline(always)]
    pub fn depth_stride(&self) -> usize {
        self.stride[DEPTH]
    }

    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.dims[ROW]
    }

    #[inline(always)]
    pub fn row_stride(&self) -> usize {
        self.stride[ROW]
    }

    #[inline(always)]
    pub fn cols(&self) -> usize {
        self.dims[COL]
    }

    #[inline(always)]
    pub fn col_stride(&self) -> usize {
        self.stride[COL]
    }

    /// Returnst the size aligned to the gguf weight memory alignment.
    ///
    /// This is useful for determining end offsets in the gguf mmap file.
    pub fn aligned_size(&self) -> usize {
        crate::buf::align(self.size(), gguf::WEIGHT_ALIGN)
    }

    /// Returns the total size of the tensor data measured in bytes.
    #[inline(always)]
    pub fn size(&self) -> usize {
        let mut n = self.dims[0] * self.stride[0] / self.typ.block_size();
        for i in 1..self.n {
            n += (self.dims[i] - 1) * self.stride[i];
        }
        n
    }

    #[inline(always)]
    pub fn set_dim(&mut self, i: usize, dim: usize) {
        self.dims[i] = dim;
    }

    #[inline(always)]
    pub fn is_transposed(&self) -> bool {
        self.stride[0] > self.stride[1]
    }

    #[inline(always)]
    pub fn is_padded_1d(&self) -> bool {
        // self.n == 1 &&
        self.stride[0] == self.typ().size()
            && self.stride[2] == self.stride[1] * self.dims[1]
            && self.stride[3] == self.stride[2] * self.dims[2]
    }

    /// A tensor is a scalar if it's 1D and has a single element.
    #[inline(always)]
    pub fn is_scalar(&self) -> bool {
        self.n == 1 && self.cols() == 1
    }

    #[inline(always)]
    pub fn is_matrix(&self) -> bool {
        self.n == 2
    }

    #[inline(always)]
    pub fn is_contiguous(&self) -> bool {
        self.stride[0] == self.typ.size()
            && self.stride[1] == (self.stride[0] * self.dims[0]) / self.typ.block_size()
            && self.stride[2] == self.stride[1] * self.dims[1]
            && self.stride[3] == self.stride[2] * self.dims[2]
    }
}

impl From<Dims> for Layout {
    #[inline(always)]
    fn from(dims: Dims) -> Self {
        Self::new_raw_dims(dims.as_slice(), Type::F32)
    }
}

impl From<D1> for Layout {
    #[inline(always)]
    fn from(dims: D1) -> Self {
        Self::new(dims, Type::F32)
    }
}

impl From<D2> for Layout {
    #[inline(always)]
    fn from(dims: D2) -> Self {
        Self::new(dims, Type::F32)
    }
}

impl From<D3> for Layout {
    #[inline(always)]
    fn from(dims: D3) -> Self {
        Self::new(dims, Type::F32)
    }
}

impl From<D4> for Layout {
    #[inline(always)]
    fn from(dims: D4) -> Self {
        Self::new(dims, Type::F32)
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}|", self.typ)?;

        let skip = MAX_DIMS - self.n;
        // NB: Print in reverse order, because dimensions and stride are stored in reverse.
        for (i, dim) in self.dims.iter().rev().skip(skip).enumerate() {
            write!(f, "{dim}")?;
            if i < self.n - 1 {
                write!(f, "x")?;
            }
        }

        write!(f, "|")?;
        for (i, stride) in self.stride.iter().rev().skip(skip).enumerate() {
            write!(f, "{stride}")?;
            if i < self.n - 1 {
                write!(f, "x")?;
            }
        }
        write!(f, "]")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_layout() {
        let layout = Layout::new(
            D3 {
                depth: 3,
                rows: 2,
                cols: 8,
            },
            Type::F32,
        );
        assert_eq!(layout.to_string(), "[F32|3x2x8|64x32x4]");
    }

    #[test]
    fn d1_and_layout_are_eq() {
        let d1 = D1 { cols: 3 };
        let layout = Layout::new(d1, Type::F16);
        let expected = Layout::new_1d(3, Type::F16);
        assert_eq!(layout.size(), expected.size());
        assert_eq!(layout.raw_dims(), expected.raw_dims());
        assert_eq!(layout.raw_stride(), expected.raw_stride());
    }
}
