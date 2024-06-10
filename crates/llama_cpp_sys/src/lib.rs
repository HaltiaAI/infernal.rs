// This is proprietary and confidential source code not for distribution.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod ggml;
pub mod llama;

pub use llama::*;

#[cfg(test)]
mod test {
    use super::*;
    use std::ffi::CStr;

    #[test]
    pub fn test_llama_print_system_info() {
        let system_info = unsafe {
            let raw_system_info = llama_print_system_info();
            CStr::from_ptr(raw_system_info)
        };
        println!("{}", system_info.to_string_lossy());
    }
}
