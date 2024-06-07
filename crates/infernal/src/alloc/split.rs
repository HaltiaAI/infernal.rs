use std::io;

use thiserror::Error;
use tracing::info;

use crate::{buf::AllocId, utils::Bytes, TensorBuilder, TensorId, TensorNotFound, TensorSrc};

/// The allocator's buffer can be lazily split up.
pub trait SplitAlloc {
    fn id(&self) -> AllocId;
    fn size(&self) -> usize;
    fn insert_new_page(&mut self, offs: usize, size: usize, pin: bool) -> io::Result<()>;
}

#[derive(Debug, Error)]
pub enum SplitError {
    #[error(transparent)]
    TensorNotFound(#[from] TensorNotFound),
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[derive(Clone, Copy, Debug)]
pub struct SplitPoint<T> {
    // Split the weight before this point.
    //
    // This is a hint, the actual page may be smaller if the memory range is not contiguous or if
    // the threshold is not on a level boundary.
    pub at: Pos<T>,
    // Pinned pages cannot be offloaded from RAM.
    pub pin: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum Pos<T> {
    Before(T),
    After(T),
}

impl<T> Pos<T> {
    pub fn inner(&self) -> &T {
        match self {
            Pos::Before(t) | Pos::After(t) => t,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Threshold {
    /// At most this percentage of the weight buffer's size.
    Percent(usize),
    /// At most this number of bytes.
    Bytes(usize),
}

impl<A: SplitAlloc, S: TensorSrc> TensorBuilder<A, S> {
    /// # Important
    ///
    /// This method assumes that the passed in tensors are at the end of level boundaries. This is
    /// not explicitly enforced though as this method is mostly used for initial testing.
    pub fn split_at_tensors(
        &mut self,
        mem_limit: usize,
        split_points: impl IntoIterator<Item = SplitPoint<TensorId>>,
    ) -> Result<(), SplitError> {
        // Tensors need to be sorted by their data offset for contiguity checks.
        let mut tensors = self.tensors.clone();
        tensors.sort_by_key(|t| t.data_offs());

        assert!(!tensors.is_empty(), "cannot split an empty tensor context");
        assert!(
            mem_limit > 0,
            "memory limit must be greater than zero, got {}",
            mem_limit
        );

        let mut split_tensors = Vec::new();
        for split in split_points {
            let split_at = split.at.inner();
            let pos = tensors
                .iter()
                .position(|t| &t.id == split_at)
                .ok_or_else(|| SplitError::TensorNotFound(TensorNotFound(*split_at)))?;
            let tensor = &tensors[pos];
            split_tensors.push((pos, tensor, split));
        }
        split_tensors.sort_by(|(_, t0, s0), (_, t1, s1)| {
            if t0.data_offs() == t1.data_offs() {
                match (s0.at, s1.at) {
                    (Pos::Before(_), Pos::After(_)) => std::cmp::Ordering::Less,
                    (Pos::After(_), Pos::Before(_)) => std::cmp::Ordering::Greater,
                    _ => panic!("split points must be unique"),
                }
            } else {
                t0.data_offs().cmp(&t1.data_offs())
            }
        });

        let mut n_splits: usize = 0;
        let mut start = 0;
        let mut page_offs = 0;
        for (pos, tensor, split) in split_tensors {
            // TODO
            let end = match split.at {
                Pos::Before(_) => pos,
                Pos::After(_) => pos + 1,
            };

            // Split point may be the first tensor, but we can't split before the first element.
            if start == end {
                continue;
            }

            debug_assert!(end <= tensors.len());

            let page_size: usize = tensors[start..end].iter().map(|t| t.aligned_size()).sum();
            assert!(page_size > 0, "split point results in empty page");
            //TODO: don't panic, instead make a smaller split
            assert!(
                page_size <= mem_limit,
                "split point exceeds memory limit: size={page_size} limit={mem_limit}"
            );

            info!(
                tensor = %tensor.name(),
                id = %tensor.id(),
                page_offs = %Bytes(page_offs),
                page_size = %Bytes(page_size),
                "splitting {} tensor",
                match split.at {
                    Pos::Before(_) => "before",
                    Pos::After(_) => "after",
                }
            );

            self.alloc
                .insert_new_page(page_offs, page_size, split.pin)?;

            let page_offs_diff = match split.at {
                Pos::Before(_) => 0,
                Pos::After(_) => tensor.aligned_size(),
            };
            page_offs = tensor.data_offs() + page_offs_diff;
            n_splits += 1;
            start = end;
        }

        // Now map the remainder of the tensors to a page, if not all tensors were mapped.
        if page_offs < self.alloc.size() {
            let page_size = self.alloc.size() - page_offs;
            info!(
                page_offs = %Bytes(page_offs),
                page_size = %Bytes(page_size),
                "allocating last split",
            );
            self.alloc.insert_new_page(page_offs, page_size, false)?;
            n_splits += 1;
        }

        assert_ne!(n_splits, 0, "no split points were found");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused)]

    use crate::{
        alloc::{AllocResult, Oom, TensorAlloc},
        buf,
        builder::{NoSrc, TensorOptionsExt},
        BufId, Layout, Type, D1,
    };

    use super::*;

    #[test]
    fn threshold_to_bytes() {
        let size = 1000;
        // assert_eq!(Threshold::Percent(50).to_bytes(size), 500);
        // assert_eq!(Threshold::Percent(100).to_bytes(size), 1000);
        // assert_eq!(Threshold::Bytes(500).to_bytes(size), 500);
        // assert_eq!(Threshold::Bytes(1000).to_bytes(size), 1000);
    }
}
