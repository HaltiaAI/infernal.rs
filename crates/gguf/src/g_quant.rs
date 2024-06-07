/// Original GGML quantization block size.
pub const BLK: usize = 32;

#[repr(C)]
pub struct BlockQ4_0 {
    pub d: u16,
    pub qs: [u8; BLK / 2],
}

#[repr(C)]
pub struct BlockQ4_1 {
    pub d: u16,
    pub m: u16,
    pub qs: [u8; BLK / 2],
}

#[repr(C)]
pub struct BlockQ5_0 {
    pub d: u16,
    pub qh: [u8; 4],
    pub qs: [u8; BLK / 2],
}

#[repr(C)]
pub struct BlockQ5_1 {
    pub d: u16,
    pub m: u16,
    pub qh: [u8; 4],
    pub qs: [u8; BLK / 2],
}

#[repr(C)]
pub struct BlockQ8_0 {
    pub d: u16,
    pub qs: [i8; BLK],
}

#[repr(C)]
pub struct BlockQ8_1 {
    pub d: f32,
    pub s: f32,
    pub qs: [i8; BLK],
}
