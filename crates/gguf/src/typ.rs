use std::mem;

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
