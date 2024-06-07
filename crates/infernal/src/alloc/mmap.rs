use std::{fs::File, io, path::Path};

use memmap2::{Advice, Mmap, MmapOptions};
use tracing::{info, trace};

use crate::{
    alloc::{split::SplitAlloc, AllocResult, AllocSize, Oom, TensorAlloc},
    backend::{BackendData, IntoBackendBufs, PagedMmap},
    buf::AllocId,
    utils::Bytes,
    BufId, Tensor,
};

/// Returns slices of the underlying memory mapped file for the requested tensor.
pub struct MmapAlloc {
    id: AllocId,
    /// The offset into the mapped buffer where the data section starts.
    data_offs: usize,
    /// Number of bytes remaining in the mapped buffer.
    free: usize,
    /// The size of the entire file. The size of the data is actually smaller.
    full_size: usize,
    // Need to keep the file around to give each page its copy.
    file: File,
    pages: Vec<(BufId, MmapPage)>,
}

#[doc(hidden)]
pub enum MmapPage {
    Pinned { offs: usize, data: Mmap },
    Offloadable(PagedMmap),
}

impl MmapPage {
    fn offs(&self) -> usize {
        match self {
            Self::Pinned { offs, .. } => *offs,
            Self::Offloadable(page) => page.offs,
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Pinned { data, .. } => data.len(),
            Self::Offloadable(page) => page.size,
        }
    }
}

impl MmapAlloc {
    #[inline(always)]
    pub fn open(path: impl AsRef<Path>, data_offs: usize) -> io::Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        Self::map(file, data_offs)
    }

    #[inline(always)]
    pub fn map(file: File, data_offs: usize) -> io::Result<Self> {
        let size = file.metadata()?.len() as usize;
        let free = size - data_offs;
        Ok(Self {
            id: AllocId::new(),
            data_offs,
            free,
            full_size: size,
            file,
            pages: Default::default(),
        })
    }

    fn data_size(&self) -> usize {
        self.full_size - self.data_offs
    }

    fn map_page(&mut self, rel_offs: usize, size: usize, pin: bool) -> io::Result<BufId> {
        // We never map the initial metadata section, it's a waste of space.
        let offs = self.data_offs + rel_offs;
        assert!(
            offs + size <= self.full_size,
            "mapping out of bounds: (offs+size)={} max={}",
            offs + size,
            self.full_size
        );
        // let size = size.min(self.data_size() - offs);

        let buf_id = BufId::new(self.id, self.pages.len() as u32);
        let page = if pin {
            info!(
                rel_offs = %Bytes(rel_offs),
                abs_offs = %Bytes(offs),
                size = %Bytes(size),
                pin,
                n_pages = self.pages.len() + 1,
                "allocating and mapping page"
            );
            MmapPage::Pinned {
                offs,
                data: mmap(&self.file, Some(offs), Some(size))?,
            }
        } else {
            info!(
                rel_offs = %Bytes(rel_offs),
                abs_offs = %Bytes(offs),
                size = %Bytes(size),
                pin,
                n_pages = self.pages.len() + 1,
                "allocating but not mapping page (yet)"
            );
            MmapPage::Offloadable(PagedMmap {
                file: self.file.try_clone()?,
                offs,
                size,
                mmap: None,
            })
        };

        self.pages.push((buf_id, page));

        Ok(buf_id)
    }
}

impl SplitAlloc for MmapAlloc {
    fn id(&self) -> AllocId {
        self.id
    }

    fn size(&self) -> usize {
        self.data_size()
    }

    fn insert_new_page(&mut self, offs: usize, size: usize, pin: bool) -> std::io::Result<()> {
        self.map_page(offs, size, pin).map(drop)
    }
}

impl AllocSize for MmapAlloc {
    #[inline(always)]
    fn alloc_size(&self) -> usize {
        self.data_size() - self.free
    }
}

pub struct MappingArgs {
    /// Offset is relative to the start of the data section. It's adjusted internally.
    pub offs: usize,
    pub size: usize,
    pub prealloc: bool,
}

impl TensorAlloc for MmapAlloc {
    type Arg = MappingArgs;

    #[inline(always)]
    fn alloc(&mut self, args: MappingArgs) -> Result<AllocResult, Oom> {
        trace!(
            size = %Bytes(args.size),
            total_so_far = %Bytes(self.alloc_size()),
            capacity = %Bytes(self.data_size()),
            free = %Bytes(self.free),
            "alloc"
        );
        if args.size > self.free {
            return Err(Oom {
                from: "mmap allocator",
                requested: args.size,
                free: self.free,
                capacity: self.data_size(),
            });
        }

        self.free -= args.size;

        let id = if args.prealloc {
            // TODO: don't ignore io::Error
            trace!(offs = %Bytes(args.offs), size = %Bytes(args.size), "preallocating page");
            self.map_page(args.offs, args.size, false)
                .map_err(|_| Oom {
                    from: "mmap allocator",
                    requested: args.size,
                    free: self.free,
                    capacity: self.data_size(),
                })?
        } else {
            BufId::unallocated()
        };

        Ok(AllocResult {
            id,
            offs: args.offs,
        })
    }

    /// Maps the weights into memory according to the split heuristic, or if that's absent, mapping
    /// the entire file in one go, and updates the tensor buffer ids accordingly.
    fn finalize_tensors(&mut self, tensors: &mut [Tensor]) -> io::Result<()> {
        // If we didn't split the mmap into pages, we map the whole file.
        if self.pages.is_empty() {
            self.map_page(0, self.data_size(), true)?;
        }

        for tensor in tensors {
            let abs_data_offs = tensor.data_offs + self.data_offs;
            // Some tensors may be preallocated.
            if !tensor.buf_id.is_allocated() {
                // Find page that contains the tensor.
                let (buf_id, page) = self
                    .pages
                    .iter()
                    .find(|(_, page)| {
                        abs_data_offs >= page.offs()
                            && abs_data_offs + tensor.size() <= page.offs() + page.size()
                    })
                    .unwrap_or_else(|| {
                        panic!(
                            "tensor '{}' buffer must be contained by mmap alloc buffer",
                            tensor.name(),
                        )
                    });
                // Update tensor buffer details.
                tensor.buf_id = *buf_id;
                // The data offset in tensor needs to be relative to the start of the page, not
                // the start of the file.
                tensor.data_offs -= page.offs() - self.data_offs;
            }
        }

        Ok(())
    }
}

impl IntoBackendBufs for MmapAlloc {
    type Data = MmapPage;
    type Iter = std::vec::IntoIter<(BufId, Self::Data)>;

    #[inline(always)]
    fn into_backend_bufs(self) -> Self::Iter {
        self.pages.into_iter()
    }
}

impl From<MmapPage> for BackendData {
    fn from(page: MmapPage) -> Self {
        match page {
            MmapPage::Pinned { data, .. } => data.into(),
            MmapPage::Offloadable(page) => page.into(),
        }
    }
}

pub(crate) fn mmap(file: &File, offs: Option<usize>, size: Option<usize>) -> io::Result<Mmap> {
    // TODO: match on OOM error here, if possible, otherwise pass in mem limit and check if file
    // is below it
    let mut opts = MmapOptions::new();
    // NB: The lib takes care of aligning the offset and size to the OS page size.
    // TODO: this does mean though that the reported mapping length won't be accurate which may be
    // a problem when being very close to memory limits. solution is to page align sizes when
    // comparing to memory limits.
    if let Some(offs) = offs {
        opts.offset(offs as u64);
    }
    if let Some(size) = size {
        opts.len(size);
    }
    // SAFETY:
    // This is safe because even though we used MAP_SHARED, the mapped pages are going to remain
    // in memory (in the kernel page cache) even if the backing file is removed, at least as
    // long as one process is referencing the pages.
    //
    // The only advantage of MAP_PRIVATE is that if another process writes to the file, we don't
    // get the updates, which is what we would want, but unfortunately it incurrs a performance
    // penalty (300 ms on an M1 Max with 400 GiB/s memory bandwidth, likely worse on the
    // iPhone), so not worth it.
    //
    // However, we can't be fully safe because the kernel's memory manager could still unmap the
    // pages when memory pressure is high (especially important on iOS) and therefore there is
    // still chance of triggering a SIGBUS when accessing pages. We attempt to avoid this by
    // mlocking the memory region but this is not a guarante AFAIK.
    let mmap = unsafe { opts.map(file) }?;
    mmap.advise(Advice::will_need())?;
    Ok(mmap)
}
