use std::mem;

use thiserror::Error;

use crate::{g_quant, k_quant, LoadError};

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TensorType {
    /// Float 32-bit.
    F32 = 0,
    /// Float 16-bit.
    F16 = 1,
    /// Quantized 4-bit (type 0).
    Q4_0 = 2,
    /// Quantized 4-bit (type 1).
    Q4_1 = 3,
    /// Quantized 5-bit (type 0).
    Q5_0 = 6,
    /// Quantized 5-bit (type 1).
    Q5_1 = 7,
    /// Quantized 8-bit (type 0).
    Q8_0 = 8,
    /// Quantized 8-bit (type 1).
    Q8_1 = 9,
    /// K-Quantized 2-bit.
    #[allow(non_camel_case_types)]
    Q2_K = 10,
    /// K-Quantized 3-bit.
    #[allow(non_camel_case_types)]
    Q3_K = 11,
    /// K-Quantized 4-bit.
    #[allow(non_camel_case_types)]
    Q4_K = 12,
    /// K-Quantized 5-bit.
    #[allow(non_camel_case_types)]
    Q5_K = 13,
    /// K-Quantized 6-bit.
    #[allow(non_camel_case_types)]
    Q6_K = 14,
    /// K-Quantized 8-bit.
    #[allow(non_camel_case_types)]
    Q8_K = 15,
    /// Integer 8-bit.
    I8 = 16,
    /// Integer 16-bit.
    I16 = 17,
    /// Integer 32-bit.
    I32 = 18,
}

impl TryFrom<u32> for TensorType {
    type Error = LoadError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        use TensorType::*;
        match value {
            0 => Ok(F32),
            1 => Ok(F16),
            2 => Ok(Q4_0),
            3 => Ok(Q4_1),
            6 => Ok(Q5_0),
            7 => Ok(Q5_1),
            8 => Ok(Q8_0),
            9 => Ok(Q8_1),
            10 => Ok(Q2_K),
            11 => Ok(Q3_K),
            12 => Ok(Q4_K),
            13 => Ok(Q5_K),
            14 => Ok(Q6_K),
            15 => Ok(Q8_K),
            16 => Ok(I8),
            17 => Ok(I16),
            18 => Ok(I32),
            _ => Err(LoadError::InvalidTensorType(value)),
        }
    }
}

impl TensorType {
    #[deprecated(note = "use size instead")]
    pub fn type_sizef(&self) -> f32 {
        self.size() as f32
    }

    #[deprecated(note = "use size instead")]
    pub fn type_size(&self) -> i32 {
        self.size() as i32
    }

    /// The size of the type's element, in bytes.
    pub fn size(&self) -> usize {
        use TensorType::*;
        match self {
            // native
            F32 => mem::size_of::<f32>(),
            F16 => mem::size_of::<f32>() / 2,
            I32 => mem::size_of::<i32>(),
            I16 => mem::size_of::<i16>(),
            I8 => mem::size_of::<i8>(),
            // original GGML quant
            Q4_0 => mem::size_of::<g_quant::BlockQ4_0>(),
            Q4_1 => mem::size_of::<g_quant::BlockQ4_1>(),
            Q5_0 => mem::size_of::<g_quant::BlockQ5_0>(),
            Q5_1 => mem::size_of::<g_quant::BlockQ5_1>(),
            Q8_0 => mem::size_of::<g_quant::BlockQ8_0>(),
            Q8_1 => mem::size_of::<g_quant::BlockQ8_1>(),
            // QK quant
            Q2_K => mem::size_of::<k_quant::BlockQ2_K>(),
            Q3_K => mem::size_of::<k_quant::BlockQ3_K>(),
            Q4_K => mem::size_of::<k_quant::BlockQ4_K>(),
            Q5_K => mem::size_of::<k_quant::BlockQ5_K>(),
            Q6_K => mem::size_of::<k_quant::BlockQ6_K>(),
            Q8_K => mem::size_of::<k_quant::BlockQ8_K>(),
        }
    }

    #[inline(always)]
    pub fn block_size(&self) -> usize {
        use TensorType::*;
        match self {
            F32 | F16 | I32 | I16 | I8 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => g_quant::BLK,
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K => k_quant::BLK,
        }
    }

    #[inline(always)]
    pub fn calc_tensor_size(&self, dims: &[u32]) -> usize {
        let mut size = self.size();
        let block_size = self.block_size();
        for dim in dims {
            size *= *dim as usize;
        }
        size / block_size
    }

    /// Returns whether this type is quantized.
    #[inline(always)]
    pub fn is_quantized(&self) -> bool {
        use TensorType::*;
        match self {
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K => {
                true
            }

            F32 | F16 | I8 | I16 | I32 => false,
        }
    }

    #[inline(always)]
    pub fn is_native(&self) -> bool {
        use TensorType::*;
        matches!(*self, F32 | I32 | I16 | I8)
    }

    #[inline(always)]
    pub fn name(&self) -> &'static str {
        use TensorType::*;
        match self {
            F32 => "f32",
            F16 => "f16",
            Q4_0 => "q4_0",
            Q4_1 => "q4_1",
            Q5_0 => "q5_0",
            Q5_1 => "q5_1",
            Q8_0 => "q8_0",
            Q8_1 => "q8_1",
            Q2_K => "q2_k",
            Q3_K => "q3_k",
            Q4_K => "q4_k",
            Q5_K => "q5_k",
            Q6_K => "q6_k",
            Q8_K => "q8_k",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
        }
    }
}

impl std::fmt::Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Denotes the GGUF file type based on its most dominant tensor type.
#[repr(u32)]
#[derive(Clone, Copy, Debug, strum::Display, strum::EnumString)]
#[strum(serialize_all = "lowercase")]
#[allow(non_camel_case_types)]
pub enum FileType {
    // LLAMA_FTYPE_ALL_F32              = 0,
    AllF32 = 0,
    // LLAMA_FTYPE_MOSTLY_F16           = 1, // except 1d tensors
    MostlyF16 = 1,
    // LLAMA_FTYPE_MOSTLY_Q4_0          = 2, // except 1d tensors
    MostlyQ4_0 = 2,
    // LLAMA_FTYPE_MOSTLY_Q4_1          = 3, // except 1d tensors
    MostlyQ4_1 = 3,
    // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
    MostlyQ4_1SomeF16 = 4,
    // Support removed:
    // MostlyQ4_2 = 5,
    // MostlyQ4_3 = 6,
    // LLAMA_FTYPE_MOSTLY_Q8_0          = 7, // except 1d tensors
    MostlyQ8_0 = 7,
    // LLAMA_FTYPE_MOSTLY_Q5_0          = 8, // except 1d tensors
    MostlyQ5_0 = 8,
    // LLAMA_FTYPE_MOSTLY_Q5_1          = 9, // except 1d tensors
    MostlyQ5_1 = 9,
    // LLAMA_FTYPE_MOSTLY_Q2_K          = 10,// except 1d tensors
    MostlyQ2_K = 10,
    // LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11,// except 1d tensors
    MostlyQ3_K_S = 11,
    // LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12,// except 1d tensors
    MostlyQ3_K_M = 12,
    // LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13,// except 1d tensors
    MostlyQ3_K_L = 13,
    // LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14,// except 1d tensors
    MostlyQ4_K_S = 14,
    // LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15,// except 1d tensors
    MostlyQ4_K_M = 15,
    // LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16,// except 1d tensors
    MostlyQ5_K_S = 16,
    // LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17,// except 1d tensors
    MostlyQ5_K_M = 17,
    // LLAMA_FTYPE_MOSTLY_Q6_K          = 18,// except 1d tensors
    MostlyQ6_K = 18,
}

impl FileType {
    pub fn guess_from_tensor(t: TensorType) -> Option<Self> {
        use TensorType::*;
        let ft = match t {
            F32 => Self::AllF32,
            F16 => Self::MostlyF16,
            Q4_0 => Self::MostlyQ4_0,
            Q4_1 => Self::MostlyQ4_1,
            Q5_0 => Self::MostlyQ5_0,
            Q5_1 => Self::MostlyQ5_1,
            Q8_0 => Self::MostlyQ8_0,
            Q2_K => Self::MostlyQ2_K,
            Q3_K => Self::MostlyQ3_K_M,
            Q4_K => Self::MostlyQ4_K_M,
            Q5_K => Self::MostlyQ5_K_M,
            Q6_K => Self::MostlyQ6_K,
            _ => return None,
        };
        Some(ft)
    }
}

#[derive(Debug, Error)]
#[error("invalid ftype: {0}")]
pub struct FileTypeConvError(u32);

impl TryFrom<u32> for FileType {
    type Error = FileTypeConvError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::AllF32),
            1 => Ok(Self::MostlyF16),
            2 => Ok(Self::MostlyQ4_0),
            3 => Ok(Self::MostlyQ4_1),
            4 => Ok(Self::MostlyQ4_1SomeF16),
            // 5 and 6 unsupported
            7 => Ok(Self::MostlyQ8_0),
            8 => Ok(Self::MostlyQ5_0),
            9 => Ok(Self::MostlyQ5_1),
            10 => Ok(Self::MostlyQ2_K),
            11 => Ok(Self::MostlyQ3_K_S),
            12 => Ok(Self::MostlyQ3_K_M),
            13 => Ok(Self::MostlyQ3_K_L),
            14 => Ok(Self::MostlyQ4_K_S),
            15 => Ok(Self::MostlyQ4_K_M),
            16 => Ok(Self::MostlyQ5_K_S),
            17 => Ok(Self::MostlyQ5_K_M),
            18 => Ok(Self::MostlyQ6_K),
            i => Err(FileTypeConvError(i)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

#[derive(Debug, Error)]
#[error("unknown token type: {0}")]
pub struct UnknownTokenType(u32);

impl TryFrom<u32> for TokenType {
    type Error = UnknownTokenType;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Undefined),
            1 => Ok(Self::Normal),
            2 => Ok(Self::Unknown),
            3 => Ok(Self::Control),
            4 => Ok(Self::UserDefined),
            5 => Ok(Self::Unused),
            6 => Ok(Self::Byte),
            _ => Err(UnknownTokenType(v)),
        }
    }
}
