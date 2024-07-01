use thiserror::Error;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Type {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

#[derive(Debug, Error)]
#[error("unknown type identifier: {0}")]
pub struct TypeConvError(u32);

impl TryFrom<u32> for Type {
    type Error = TypeConvError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            i if i == Type::Uint8 as u32 => Ok(Type::Uint8),
            i if i == Type::Int8 as u32 => Ok(Type::Int8),
            i if i == Type::Uint16 as u32 => Ok(Type::Uint16),
            i if i == Type::Int16 as u32 => Ok(Type::Int16),
            i if i == Type::Uint32 as u32 => Ok(Type::Uint32),
            i if i == Type::Int32 as u32 => Ok(Type::Int32),
            i if i == Type::Float32 as u32 => Ok(Type::Float32),
            i if i == Type::Bool as u32 => Ok(Type::Bool),
            i if i == Type::String as u32 => Ok(Type::String),
            i if i == Type::Array as u32 => Ok(Type::Array),
            i if i == Type::Uint64 as u32 => Ok(Type::Uint64),
            i if i == Type::Int64 as u32 => Ok(Type::Int64),
            i if i == Type::Float64 as u32 => Ok(Type::Float64),
            i => Err(TypeConvError(i)),
        }
    }
}

#[derive(Debug)]
pub enum Value {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(Vec<u8>),
    Array(Vec<Value>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Error)]
#[error("failed to cast gguf value to {exp}, value is {got}")]
pub struct TypeCastError {
    exp: &'static str,
    got: &'static str,
}

macro_rules! value_casts {
    ($($f:ident => $v:ident($t:ty),)*) => {
        impl Value {
        $(
            pub fn $f(&self) -> Result<$t, TypeCastError> {
                if let Self::$v(v) = self {
                    Ok(*v)
                } else {
                    Err(TypeCastError {
                        exp: Self::$v(Default::default()).type_str(),
                        got: self.type_str(),
                    })
                }
            }

        )*
        }
    };
}

value_casts!(
    as_bool => Bool(bool),
    as_u8 => Uint8(u8),
    as_i8 => Int8(i8),
    as_u16 => Uint16(u16),
    as_i16 => Int16(i16),
    as_u32 => Uint32(u32),
    as_i32 => Int32(i32),
    as_f32 => Float32(f32),
    as_f64 => Float64(f64),
    // as_bytestring => String(&[u8]),
    // as_array => Array(&[Value]),
);

impl Value {
    pub fn as_bytestring(&self) -> Result<&[u8], TypeCastError> {
        if let Self::String(s) = self {
            Ok(s.as_slice())
        } else {
            Err(TypeCastError {
                exp: Self::String(Default::default()).type_str(),
                got: self.type_str(),
            })
        }
    }

    pub fn into_bytestring(self) -> Result<Vec<u8>, TypeCastError> {
        if let Self::String(s) = self {
            Ok(s)
        } else {
            Err(TypeCastError {
                exp: Self::String(Default::default()).type_str(),
                got: self.type_str(),
            })
        }
    }

    pub fn as_array(&self) -> Result<&[Value], TypeCastError> {
        if let Self::Array(s) = self {
            Ok(s)
        } else {
            Err(TypeCastError {
                exp: Self::Array(Default::default()).type_str(),
                got: self.type_str(),
            })
        }
    }

    pub fn as_array_len(&self) -> Result<usize, TypeCastError> {
        if let Self::Array(s) = self {
            Ok(s.len())
        } else {
            Err(TypeCastError {
                exp: Self::Array(Default::default()).type_str(),
                got: self.type_str(),
            })
        }
    }

    pub fn type_str(&self) -> &'static str {
        match self {
            Self::Uint8(_) => "u8",
            Self::Int8(_) => "i8",
            Self::Uint16(_) => "u16",
            Self::Int16(_) => "i16",
            Self::Uint32(_) => "u32",
            Self::Int32(_) => "i32",
            Self::Float32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::Uint64(_) => "u64",
            Self::Int64(_) => "i64",
            Self::Float64(_) => "f64",
        }
    }
}

#[derive(Debug)]
pub struct Kv {
    pub key: Vec<u8>,
    pub value: Value,
    // typ: Type, TODO: is this needed? value already contains the info
}

/// LLM related keys for GGUF's key-value store.
#[derive(Clone, Copy, Debug)]
pub enum KvKey {
    GeneralArchitecture,
    GeneralQuantizationVersion,
    GeneralAlignment,
    GeneralName,
    GeneralAuthor,
    GeneralUrl,
    GeneralDescription,
    GeneralLicense,
    GeneralSourceUrl,
    GeneralSourceHfRepo,
    GeneralFileType,

    ContextLength,
    EmbeddingLength,
    BlockCount,
    FeedForwardLength,
    UseParallelResidual,
    TensorDataLayout,

    AttentionHeadCount,
    AttentionHeadCountKv,
    AttentionMaxAlibiBias,
    AttentionClampKqv,
    AttentionLayernormEps,
    AttentionLayernormRmsEps,

    RopeDimensionCount,
    RopeFreqBase,
    RopeScaleLinear,

    TokenizerModel,
    TokenizerList,
    TokenizerTokenType,
    TokenizerScores,
    TokenizerMerges,
    TokenizerBosId,
    TokenizerEosId,
    TokenizerUnkId,
    TokenizerSepId,
    TokenizerPadId,
    TokenizerHfJson,
    TokenizerRwkv,
}
