use std::{
    alloc::{self, Layout},
    ops::{Deref, DerefMut},
    os::raw::c_void,
    ptr::NonNull,
};

use libc::_SC_PAGESIZE;

/// Fixed size page aligned buffer that can be used in Metal buffers (which requires page alignment).
#[derive(Debug)]
pub struct PageAlignedBuf {
    data: NonNull<c_void>,
    layout: Layout,
}

impl PageAlignedBuf {
    /// Creates a new buffer of the specified size aligned to the OS's page size (_SC_PAGESIZE).
    /// This is important for Metal performance. TODO: add links
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "buffer size must be larger than zero");
        let align = unsafe { libc::sysconf(_SC_PAGESIZE) };
        let layout = Layout::from_size_align(size, align as usize).unwrap();

        unsafe {
            PageAlignedBuf {
                data: NonNull::new(alloc::alloc(layout).cast())
                    .expect("failed to alloce {size} bytes"),
                layout,
            }
        }
    }

    /// Returns the size of the buffer in bytes
    #[allow(clippy::len_without_is_empty)] // not possible for buffer to be empty
    pub fn len(&self) -> usize {
        self.layout.size()
    }
}

impl Drop for PageAlignedBuf {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.data.as_ptr().cast(), self.layout);
        }
    }
}

impl Deref for PageAlignedBuf {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: This is safe because only the constructor allocates memory and from that point on
        // the memory is never touched until deallocation.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast(), self.layout.size()) }
    }
}

impl DerefMut for PageAlignedBuf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: See above.
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr().cast(), self.layout.size()) }
    }
}
