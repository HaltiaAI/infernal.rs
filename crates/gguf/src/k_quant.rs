/// QK quantization with K=256.
pub const BLK: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

#[repr(C)]
pub struct BlockQ2_K {
    pub scales: [u8; BLK / 16],
    pub qs: [u8; BLK / 4],
    pub d: u16,
    pub dmin: u16,
}

#[repr(C)]
pub struct BlockQ3_K {
    pub hmask: [u8; BLK / 8],
    pub qs: [u8; BLK / 4],
    pub scales: [u8; 12],
    pub d: u16,
}

#[repr(C)]
pub struct BlockQ4_K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; BLK / 2],
}

#[repr(C)]
pub struct BlockQ5_K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; BLK / 8],
    pub qs: [u8; BLK / 2],
}

#[repr(C)]
pub struct BlockQ6_K {
    pub ql: [u8; BLK / 2],
    pub qh: [u8; BLK / 4],
    pub scales: [i8; BLK / 16],
    pub d: u16,
}

#[repr(C)]
pub struct BlockQ8_K {
    pub d: f32,
    pub qs: [i8; BLK],
    pub bsums: [i16; BLK / 16],
}

#[test]
fn k_quant_tensor_type_sizes_match_ggml() {
    assert_eq!(crate::TensorType::Q2_K.size(), 84);
    assert_eq!(crate::TensorType::Q3_K.size(), 110);
    assert_eq!(crate::TensorType::Q4_K.size(), 144);
    assert_eq!(crate::TensorType::Q5_K.size(), 176);
    assert_eq!(crate::TensorType::Q6_K.size(), 210);
}
