use std::{collections::HashSet, time::Instant, sync::Arc, ops::Deref};

use tracing::{debug, info, trace, warn};

use crate::{
    backend::{BackendBuf, BufStatus, DanglingTensor, MemoryLimitTooLow},
    utils::Bytes,
    BufId, Tensor,
};

pub(crate) struct DeviceInfo {
    pub mem_limit: usize,
    pub max_buf_size: usize,
}

pub(crate) trait Device {
    type Buf;

    fn info(&self) -> DeviceInfo;
    fn new_buf(&self, buf: &BackendBuf) -> std::io::Result<Self::Buf>;
    fn free_buf(&self, buf: Self::Buf) -> std::io::Result<()>;

    fn has_noncopy_bufs(&self) -> bool {
        false
    }

    fn align(&self, size: usize) -> usize {
        size
    }
}

pub(crate) struct PageCache<D: Device> {
    /// Ordered by size, from smallest to largest.
    pages: Vec<Page<D>>,

    mem_limit: usize,
    device_info: DeviceInfo,

    /// Buffers that are staged to be queued for execution. This is a 2-step process because it may
    /// be that the newly staged buffers would cause us to exceed the configured memory limit, in
    /// which case we need to execute the queued buffers and queue these later.
    staging_queue: HashSet<BufId>,
    /// Buffers that are currently queued for execution, which may or may not already be loaded into
    /// VRAM.
    ///
    /// Buffers in the queue must not be offloaded.
    exec_queue: HashSet<BufId>,

    /// Total size of buffers that are currently paged into VRAM.
    vram_use: usize,
    /// The size of the queued buffers that are *not* loaded into VRAM yet. This is effectively
    /// a staging ground for buffers so we can check that maximum memory use is within limit before
    /// loading buffers onto the GPU.
    additional_vram_use: usize,

    /// Total size of buffers that can't be offloaded from memory.
    fixed_ram_use: usize,
    /// Total size of buffers that are currently paged into memory, only counting buffers that are
    /// offloadable.
    dyn_ram_use: usize,
}

impl<D: Device> PageCache<D> {
    pub fn new(mem_limit: usize, device: &D) -> Self {
        Self {
            pages: Vec::new(),
            mem_limit,
            device_info: device.info(),
            staging_queue: HashSet::new(),
            exec_queue: HashSet::new(),
            vram_use: 0,
            additional_vram_use: 0,
            fixed_ram_use: 0,
            dyn_ram_use: 0,
        }
    }

    #[inline(always)]
    pub fn mem_limit(&self) -> usize {
        self.mem_limit
    }

    pub fn set_mem_limit(&mut self, mem_limit: usize) -> Result<(), MemoryLimitTooLow> {
        if mem_limit == self.mem_limit {
            return Ok(());
        }

        assert!(
            mem_limit <= self.device_info.mem_limit,
            "memory limit exceeds device memory limit"
        );

        // Verify that the new value fits all currently configured buffers.
        for page in &self.pages {
            let size = page.buf.size();
            if size > mem_limit {
                return Err(MemoryLimitTooLow {
                    needed: size,
                    limit: mem_limit,
                });
            }
        }

        self.mem_limit = mem_limit;

        warn!(
            mem_limit = %Bytes(mem_limit),
            ram_use = %Bytes(self.ram_use()),
            vram_use = %Bytes(self.vram_use),
            "set new memory limit"
        );

        Ok(())
    }

    pub fn insert(&mut self, buf: BackendBuf) -> Result<(), MemoryLimitTooLow> {
        let id = buf.id();
        let size = buf.size();

        if self.pages.iter().any(|p| p.buf.id == id) {
            // TODO: error
            panic!("buffer already in page cache");
        }

        if size > self.mem_limit {
            return Err(MemoryLimitTooLow {
                needed: size,
                limit: self.mem_limit,
            });
        }

        assert!(
            size <= self.device_info.max_buf_size,
            "sharding buffers into views is not yet implemented"
        );

        // If buffer is loaded and it exceeds total RAM limit (TODO: not currently impl'd), offload
        // buffer immediately.
        if buf.status() == BufStatus::Loaded && self.ram_use() + size > self.mem_limit {
            // TODO
        }

        if !buf.is_offloadable() {
            self.fixed_ram_use += buf.size();
        } else if buf.status() == BufStatus::Loaded {
            self.dyn_ram_use += buf.size();
        }

        // Find the insert position. The pages list is ordered by ascending order of page size so we
        // can use binary search.
        let pos = self
            .pages
            .binary_search_by_key(&size, |page| page.buf.size())
            .unwrap_or_else(|x| x);
        self.pages.insert(pos, Page::new(buf));

        info!(
            id = %id,
            size = %Bytes(size),
            name = %self.pages[pos].buf.name,
            pos,
            pages = ?self.pages
                .iter()
                .map(|p| (p.buf.name.as_ref(), Bytes(p.buf.size())))
                .collect::<Vec<_>>(),
            "inserted buffer into page cache"
        );

        Ok(())
    }

    pub fn get(&self, id: BufId) -> Option<&BackendBuf> {
        self.get_page(id).map(|page| &page.buf)
    }

    fn get_page(&self, id: BufId) -> Option<&Page<D>> {
        self.pages.iter().find(|p| p.buf.id == id)
    }

    pub fn ram_use(&self) -> usize {
        self.fixed_ram_use + self.dyn_ram_use
    }

    /// Returns the size of current GPU buffers plus the additional buffers in the queue expected
    /// to be mapped into GPU.
    pub fn expected_vram_use(&self) -> usize {
        self.vram_use + self.additional_vram_use
    }

    pub fn vram_use(&self) -> usize {
        self.vram_use
    }

    pub fn stage(&mut self, device: &D, id: BufId) -> usize {
        // Only stage buffer if it's not been staged or queued for execution yet.
        if !self.exec_queue.contains(&id) && self.staging_queue.insert(id) {
            let page = self.get_page(id).unwrap(); // TODO
            let size = device.align(page.buf.size());
            debug!(
                %id,
                size = %Bytes(size),
                name = %page.buf.name,
                "staged buffer for execution"
            );

            if page.device_buf.is_none() {
                trace!(
                    %id,
                    size = %Bytes(size),
                    name = %page.buf.name,
                    additional_vram_use = %Bytes(self.additional_vram_use + size),
                    "buffer not yet on GPU"
                );
                self.additional_vram_use += size;
            }
            size
        } else {
            0
        }
    }

    pub fn clear_exec_queue(&mut self) {
        debug!(len = self.exec_queue.len(), "clearing execution queue");
        self.exec_queue.clear();
    }

    pub fn clear_staging_queue(&mut self) {
        debug!(len = self.staging_queue.len(), "clearing staging queue");
        self.staging_queue.clear();
    }

    /// Attempts to make space for the staging queue by freeing old unused buffers (that are not in
    /// the staging and execution queues).
    pub fn free_space(&mut self, device: &D) -> std::io::Result<()> {
        if self.expected_vram_use() <= self.mem_limit {
            return Ok(());
        }

        let start = Instant::now();

        trace!(
            vram_use = %Bytes(self.vram_use),
            additional_vram_use = %Bytes(self.additional_vram_use),
            vram_limit = %Bytes(self.mem_limit),
            ram_use = %Bytes(self.ram_use()),
            "freeing space for buffers in queue",
        );

        let prev_ram_use = self.ram_use();
        let prev_vram_use = self.vram_use;

        // Find the smallest buffer to offload that would allow the new queue to fit within the
        // memory limit. Note that pages are ordered in ascending order of size.
        // TODO(perf): bench: prioritize unmap as few buffers (regardless of size) or total size unmapped?
        for i in 0..self.pages.len() {
            if self.expected_vram_use() > self.mem_limit {
                let page = &self.pages[i];
                let id = page.buf.id;
                // Only offload buffers that are not staged or queued for execution.
                if page.device_buf.is_some()
                    && !self.staging_queue.contains(&id)
                    && !self.exec_queue.contains(&id)
                {
                    self.offload(device, id)?;
                }
            } else {
                break;
            }
        }

        let ram_freed = prev_ram_use - self.ram_use();
        let vram_freed = prev_vram_use - self.vram_use;
        if ram_freed > 0 || vram_freed > 0 {
            trace!(
                time = ?start.elapsed(),
                vram_freed = %Bytes(vram_freed),
                vram_use = %Bytes(self.vram_use),
                additional_vram_use = %Bytes(self.additional_vram_use),
                vram_limit = %Bytes(self.mem_limit),
                ram_freed = %Bytes(ram_freed),
                ram_use = %Bytes(self.ram_use()),
                "freed space for buffers in queue",
            );
        } else {
            debug!(
                time = ?start.elapsed(),
                "no space freed for staged buffers, cannot execute",
            );
        }

        Ok(())
    }

    pub fn load(&mut self, device: &D, id: BufId) -> std::io::Result<()> {
        let page = self.pages.iter_mut().find(|p| p.buf.id == id).unwrap(); // TODO
                                                                            //
        trace!(id = %id, %page.buf.name, "loading page");

        let start = Instant::now();

        // Load the buffer into RAM first, if not loaded.
        let ram_size = page.load_into_ram()?;
        self.dyn_ram_use += ram_size;
        let ram_load_time = start.elapsed();

        // Then load the buffer into VRAM if it's not already there.
        let vram_size = if page.device_buf.is_none() {
            let size = page.load_into_vram(device)?;
            self.vram_use += size;
            self.additional_vram_use -= size;
            size
        } else {
            0
        };
        let time = start.elapsed();
        let vram_load_time = time - ram_load_time;

        // Remove buffer from the staging queue...
        self.staging_queue.remove(&id);
        // ...and add it to the execution queue, to prevent it from being unmapped by free_space.
        self.exec_queue.insert(id);

        if ram_size > 0 || vram_size > 0 {
            debug!(
                total_time = ?start.elapsed(),
                ?ram_load_time,
                ?vram_load_time,
                %id,
                name = %page.buf.name,
                ram_size = %Bytes(ram_size),
                vram_size = %Bytes(vram_size),
                "loaded page"
            );
        }

        Ok(())
    }

    pub fn offload(&mut self, device: &D, id: BufId) -> std::io::Result<()> {
        let page = self.pages.iter_mut().find(|p| p.buf.id == id).unwrap(); // TODO

        trace!(id = %id, %page.buf.name, "offloading page");

        // Remove buffer from VRAM first if it's loaded.
        if page.device_buf.is_some() {
            let size = page.offload_from_vram(device)?;
            self.vram_use -= size;
        }

        // Remove buffer from RAM, if applicable.
        self.dyn_ram_use -= page.offload_from_ram(device)?;

        trace!(id = %id, %page.buf.name, "offloaded page");

        Ok(())
    }

    pub fn get2(&self, t0: &Tensor, t1: &Tensor) -> Result<[AllocatedPage<D>; 2], DanglingTensor> {
        let page0 = self.get_page(t0.buf_id).ok_or(DanglingTensor::new(t0))?;

        let page1 = if t1.buf_id == t0.buf_id {
            page0
        } else {
            self.get_page(t1.buf_id).ok_or(DanglingTensor::new(t1))?
        };

        Ok([
            AllocatedPage {
                buf: &page0.buf,
                device_buf: page0.device_buf.as_ref().expect(UNPAGED_BUG_MSG),
            },
            AllocatedPage {
                buf: &page1.buf,
                device_buf: page1.device_buf.as_ref().expect(UNPAGED_BUG_MSG),
            }
        ])
    }

    pub fn get3(
        &self,
        t0: &Tensor,
        t1: &Tensor,
        t2: &Tensor,
    ) -> Result<[&D::Buf; 3], DanglingTensor> {
        let page0 = self.get_page(t0.buf_id).ok_or(DanglingTensor::new(t0))?;

        let page1 = if t1.buf_id == t0.buf_id {
            page0
        } else {
            self.get_page(t1.buf_id).ok_or(DanglingTensor::new(t1))?
        };

        let page2 = if t2.buf_id == t0.buf_id {
            page0
        } else {
            self.get_page(t2.buf_id).ok_or(DanglingTensor::new(t2))?
        };

        Ok([
            page0.device_buf.as_ref().expect(UNPAGED_BUG_MSG),
            page1.device_buf.as_ref().expect(UNPAGED_BUG_MSG),
            page2.device_buf.as_ref().expect(UNPAGED_BUG_MSG),
        ])
    }
}

const UNPAGED_BUG_MSG: &str = "Backend buffer not paged in. This is a bug, please report.";

pub(crate) struct AllocatedPage<'a, D: Device> {
    pub buf: &'a BackendBuf,
    pub device_buf: &'a D::Buf,
}

impl<'a, D: Device> Deref for AllocatedPage<'a, D> {
    type Target = D::Buf;

    fn deref(&self) -> &'a Self::Target {
        self.device_buf
    }
}

struct Page<D: Device> {
    buf: BackendBuf,
    device_buf: Option<D::Buf>,
}

impl<D: Device> Page<D> {
    fn new(buf: BackendBuf) -> Self {
        Self {
            buf: buf,
            device_buf: None,
        }
    }

    fn load_into_ram(&mut self) -> std::io::Result<usize> {
        if self.buf.status() == BufStatus::Offloaded {
            // NB: This borrows the rc but it's safe because while backend is executing the graph no
            // one has access to these buffers. It's an API error to keep a buffer borrowed while
            // calling compute_graph.
            let size = self.buf.size();
            trace!(name = %self.buf.name, id = %self.buf.id, size = %Bytes(size), "loading buffer into RAM");
            self.buf.load()?;
            debug!(name = %self.buf.name, id = %self.buf.id, size = %Bytes(size), "loaded buffer into RAM");
            Ok(size)
        } else {
            Ok(0)
        }
    }

    fn offload_from_ram(&mut self, device: &D) -> std::io::Result<usize> {
        if self.device_buf.is_some() && device.has_noncopy_bufs() {
            panic!("cannot offload buffer from RAM if GPU buffer references this buffer");
        }
        if self.buf.status() == BufStatus::Loaded && self.buf.is_offloadable() {
            let size = self.buf.size();
            trace!(name = %self.buf.name, id = %self.buf.id, size = %Bytes(size), "offloading buffer from RAM");
            let new_status = self.buf.offload();
            debug!(name = %self.buf.name, id = %self.buf.id, size = %Bytes(size), ?new_status, "offloaded buffer from RAM");
            Ok(size)
        } else {
            Ok(0)
        }
    }

    fn load_into_vram(&mut self, device: &D) -> std::io::Result<usize> {
        if self.device_buf.is_none() {
            let size = device.align(self.buf.size());
            trace!(name = %self.buf.name, id = %self.buf.id, size = %Bytes(size), "creating GPU buffer");
            assert!(
                self.buf.status() == BufStatus::Loaded,
                "buffer cannot be loaded into VRAM if not in RAM"
            );
            self.device_buf = Some(device.new_buf(&self.buf)?);
            debug!(name = %self.buf.name, id = %self.buf.id, size = %Bytes(size), "created GPU buffer");
            Ok(size)
        } else {
            Ok(0)
        }
    }

    fn offload_from_vram(&mut self, device: &D) -> std::io::Result<usize> {
        if let Some(buf) = self.device_buf.take() {
            let size = device.align(self.buf.size());
            trace!(name = %self.buf.name, id = %self.buf.id, size = %Bytes(size), "freeing GPU buffer from VRAM");
            device.free_buf(buf)?;
            debug!(name = %self.buf.name, id = %self.buf.id, size = %Bytes(size), "freed GPU buffer from VRAM");
            Ok(size)
        } else {
            Ok(0)
        }
    }
}
