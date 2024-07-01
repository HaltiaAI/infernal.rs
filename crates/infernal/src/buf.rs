use core::fmt;
use std::{
    alloc,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::atomic::{AtomicIsize, Ordering},
};

use libc::_SC_PAGESIZE;

use crate::alloc::Oom;

/// A buffer type that has a [`BufId`] but doesn't have any underlying data.
///
/// This is useful for types that expect a buffer but don't write to it, such as measure allocators
/// like [`GraphAlloc::measure`](crate::alloc::graph::GraphAlloc::measure).
///
/// This type has a unique but non-constructable buffer id.
pub struct PhantomBuf;

impl Deref for PhantomBuf {
    type Target = [u8];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &[]
    }
}
impl DerefMut for PhantomBuf {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut []
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocId(isize);

impl AllocId {
    pub(crate) fn new() -> Self {
        static NEXT: AtomicIsize = AtomicIsize::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    pub(crate) fn from_addr(b: &[u8]) -> Self {
        Self(b.as_ptr() as _)
    }

    const fn phantom() -> AllocId {
        Self(-2)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufId {
    /// The allocator the buffer belongs to.
    alloc: AllocId,
    /// The index of the buffer within the allocator. Most allocators only have a single buffer.
    idx: i32,
}

impl Default for BufId {
    fn default() -> Self {
        Self::unallocated()
    }
}

impl BufId {
    pub fn is_allocated(&self) -> bool {
        (self.alloc == AllocId::phantom() || self.alloc.0 >= 0) && self.idx >= 0
    }

    pub(crate) fn alloc_id(&self) -> AllocId {
        self.alloc
    }

    pub(crate) fn idx(&self) -> i32 {
        self.idx
    }

    pub(crate) fn new(alloc: AllocId, idx: u32) -> Self {
        Self {
            alloc,
            idx: idx as i32,
        }
    }

    /// This is the default state.
    pub const fn unallocated() -> Self {
        Self {
            alloc: AllocId(-1),
            idx: -1,
        }
    }

    /// Phantom buffer has a specific meaning, it's tied to a [`PhantomBuf`] (similar to
    /// `PhantomData`) that acts as a buffer but is not actually allocated. The id thus needs to be
    /// different from an unallocated buffer in scenarios when specifically want to know the buffer
    /// is unallocated (e.g. when using a measure allocator in which the phantom buffer pretends to
    /// be a real buffer).
    pub(crate) const fn phantom() -> Self {
        Self {
            alloc: AllocId::phantom(),
            // We simulate a single buffer allocator. This is necessary because if we leave this at
            // n < 0, the buffer will be considered unallocated by `Self::is_allocated`.
            idx: 0,
        }
    }

    pub(crate) fn from_addr(b: &[u8]) -> Self {
        Self {
            alloc: AllocId::from_addr(b),
            idx: 0,
        }
    }
}

impl fmt::Display for BufId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "buf[{}#{}]", self.alloc.0, self.idx)
    }
}

pub(crate) fn align_buf(buf: &mut [u8], align: usize) -> &mut [u8] {
    let align = aligned_offset(buf.as_ptr() as _, 0, align);
    let ptr = unsafe { buf.as_mut_ptr().add(align) };
    let len = buf.len() - align;
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

pub(crate) fn align(t: usize, align: usize) -> usize {
    aligned_offset(std::ptr::null::<usize>(), t, align)
}

// Direct 1 to 1 translation of the ggml function, hence the awkwardness.
pub(crate) fn aligned_offset(ptr: *const usize, offs: usize, align: usize) -> usize {
    assert!(align.is_power_of_two());
    let align_offs = (align - ((ptr as usize + offs) % align)) % align;
    debug_assert_eq!((ptr as usize + offs + align_offs) % align, 0);
    offs + align_offs
}

pub(crate) fn get_page_size() -> usize {
    unsafe { libc::sysconf(_SC_PAGESIZE) as usize }
}

/// Fixed size page aligned buffer that can be used in Metal buffers (which requires page alignment).
#[derive(Debug)]
pub struct Buf {
    data: NonNull<u8>,
    layout: alloc::Layout,
}

impl Buf {
    /// Creates a new buffer of the specified size aligned to the OS's page size (_SC_PAGESIZE).
    /// This is important for Metal performance. TODO: add links
    pub fn new(size: usize) -> Result<Self, Oom> {
        assert!(size > 0, "buffer size must be larger than zero");
        let align = get_page_size();
        // TODO: consider using above align functions to get the total allocated size (may not equal
        // usable size though, keep in mind)
        let layout = alloc::Layout::from_size_align(size, align).unwrap();

        unsafe {
            Ok(Buf {
                data: NonNull::new(alloc::alloc(layout)).ok_or(Oom {
                    from: "OS",
                    requested: size,
                    free: 0,
                    capacity: 0,
                })?,
                layout,
            })
        }
    }

    /// Returns the size of the buffer in bytes
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.layout.size()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.data.as_ptr(), self.layout);
        }
    }
}

impl Deref for Buf {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: This is safe because only the constructor allocates memory and from that point on
        // the memory is never touched until deallocation.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.layout.size()) }
    }
}

impl DerefMut for Buf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: See above.
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.layout.size()) }
    }
}

// #[test]
// fn alignment_matches_ggml() {
//     const ALIGNMENT: usize = 32;
//     let mut buf = Buf::new(1052).unwrap();
//     let offs = 5;
//     unsafe {
//         let ggml_offs = llama_cpp_sys::ggml::aligned_offset(buf.as_mut_ptr() as _, offs, ALIGNMENT);
//         let our_offs = aligned_offset(buf.as_mut_ptr() as _, offs, ALIGNMENT);
//         assert_eq!(ggml_offs, our_offs);

//         for t in [0, 5, 1952, 13, 32, 191] {
//             let ggml_offs = llama_cpp_sys::ggml::aligned_offset(std::ptr::null_mut(), t, ALIGNMENT);
//             let our_offs = align(t, ALIGNMENT);
//             assert_eq!(ggml_offs, our_offs, "offset {t} failed");
//         }
//     }
// }

pub trait SliceCast {
    fn cast<U>(&self) -> &[U];
    fn cast_mut<U>(&mut self) -> &mut [U];
}

impl<T> SliceCast for [T] {
    #[inline(always)]
    fn cast<U>(&self) -> &[U] {
        assert_slice_alignment::<T, U>(self);
        let len = std::mem::size_of_val(self) / std::mem::size_of::<U>();
        let ptr = self.as_ptr() as *const U;
        // SAFETY: This is safe because we know that the buffer length is a multiple of the target
        // type size.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    #[inline(always)]
    fn cast_mut<U>(&mut self) -> &mut [U] {
        assert_slice_alignment::<T, U>(self);
        let len = std::mem::size_of_val(self) / std::mem::size_of::<U>();
        let ptr = self.as_ptr() as *mut U;
        // SAFETY: This is safe because we know that the buffer length is a multiple of the target
        // type size.
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

/// Asserts that the alignment of the source pointer is compatible with the alignment of U.
fn assert_slice_alignment<T, U>(src: &[T]) {
    assert_eq!(
        (src.as_ptr() as usize) % std::mem::align_of::<U>(),
        0,
        "The source slice's alignment is not compatible with the destination type."
    );
}

#[test]
fn casts_to_correct_slice_cast_len() {
    let src: &[i32] = &[1, 2, 3, 4, 5, 6];
    let dst: &[i16] = src.cast();
    assert_eq!(dst.len(), src.len() * 2);
}
