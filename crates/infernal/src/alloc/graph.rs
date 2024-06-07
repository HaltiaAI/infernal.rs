use std::{
    collections::HashMap,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use tracing::trace;

use crate::{
    alloc::{AllocResult, AllocSize, Oom, ReadWriteAlloc, TensorAlloc},
    buf::{self, PhantomBuf},
    graph::opt::ConcurLevel,
    op::{BinaryOp, Op},
    tensor::CtxId,
    BufId, ComputeGraph, Tensor, TensorId,
};

#[derive(Default)]
pub struct RawParts {
    free_blocks: Vec<FreeBlock>,
    nodes: HashMap<TensorId, Node>,
    max_mem_use: usize,
}

pub struct GraphAlloc<B> {
    id: BufId,
    data: B,
    free_blocks: Vec<FreeBlock>,
    nodes: HashMap<TensorId, Node>,
    max_mem_use: usize,
}

// TODO: remove debug
#[derive(Debug, Default)]
struct Node {
    n_children: usize,
    n_views: usize,
}

impl Node {
    fn has_unique_child(&self) -> bool {
        self.n_children == 1 && self.n_views == 0
    }

    fn has_unique_view(&self) -> bool {
        self.n_children == 0 && self.n_views == 1
    }

    fn is_unreferenced(&self) -> bool {
        self.n_children == 0 && self.n_views == 0
    }
}

#[derive(Debug)]
struct FreeBlock {
    offs: usize,
    free: usize,
}

impl FreeBlock {
    fn end_offs(&self) -> usize {
        self.offs + self.free
    }
}

impl GraphAlloc<PhantomBuf> {
    #[inline(always)]
    pub fn measure() -> Self {
        Self {
            id: BufId::phantom(),
            data: PhantomBuf,
            free_blocks: vec![FreeBlock {
                // TODO: should we offset this to ALIGNMENT - 1 to account for worst case buffer
                // offset? This should almost always be just the buffer given to us by the system
                // allocator which is automatically aligned, so seems superfluous.
                offs: 0,
                // Simulate infinite memory.
                free: usize::MAX,
            }],
            nodes: Default::default(),
            max_mem_use: 0,
        }
    }
}

impl<'a> GraphAlloc<&'a mut [u8]> {
    #[inline(always)]
    pub fn new(buf: &'a mut [u8]) -> Self {
        // Align slice to
        Self {
            id: BufId::from_addr(buf),
            data: align_buf(buf),
            free_blocks: Default::default(),
            nodes: Default::default(),
            max_mem_use: 0,
        }
    }

    /// Reuse some parts of a previously used allocator to reuse allocated memory and resets it.
    #[inline(always)]
    pub fn with_raw_parts(mut self, parts: RawParts) -> Self {
        self.free_blocks = parts.free_blocks;
        self.nodes = parts.nodes;
        self.max_mem_use = parts.max_mem_use;
        self.reset();
        self
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.clear();
        self.free_blocks.push(FreeBlock {
            offs: 0,
            free: self.data.len(),
        });
    }
}

impl<B: Deref<Target = [u8]>> GraphAlloc<B> {
    #[inline(always)]
    pub fn into_raw_parts(self) -> RawParts {
        RawParts {
            free_blocks: self.free_blocks,
            nodes: self.nodes,
            max_mem_use: self.max_mem_use,
        }
    }

    /// Preallocates internal buffers for the expected number of nodes.
    ///
    /// Example of which models use how many blocks:
    /// - Llama 7B: 1124
    pub fn with_expected_nodes(mut self, n: usize) -> Self {
        if self.nodes.capacity() < n {
            self.nodes.reserve(n - self.nodes.capacity());
        }
        self
    }

    /// Preallocates internal buffers for the expected number of allocation blocks.
    ///
    /// Example of which models use how many blocks:
    /// - Llama 7B: TODO (46)
    pub fn with_expected_blocks(mut self, n: usize) -> Self {
        if self.free_blocks.capacity() < n {
            self.free_blocks.reserve(n - self.free_blocks.capacity());
        }
        self
    }

    /// Optimally allocates all tensors in the graph with maximum memory reuse and returns the
    /// maximum memory usage of the graph.
    ///
    /// If this is a measure allocator, the maximum memory usage can then be used to instantiate
    /// a buffer (and thus only allocate as much memory as is actually used) and passed into
    /// [`Self::new`] or [`Self::reuse`].
    ///
    /// # Important
    ///
    /// The allocator assumes that all tensors passed in are unallocated and their data is going to be
    /// allocated here. The source tensors of these tensors (e.g. leaves) need *not* be present.
    pub fn alloc(
        &mut self,
        graph: &ComputeGraph,
        mem_limit: usize,
        tensors: &mut [Tensor],
        ctx_id: CtxId,
    ) -> Result<usize, Oom> {
        let exec_plan = graph
            .exec_plan(mem_limit)
            .expect("graph execution plan must be optimized before allocation");

        let mut ctx = Ctx {
            graph,
            tensors,
            ctx_id,
        };

        self.count_nodes(&ctx);

        for (i, level) in exec_plan.levels.iter().enumerate() {
            // Allocate nodes in this level.
            for node_idx in level {
                let id = graph.nodes()[*node_idx];
                let Some(tensor) = ctx.get(id) else { continue };
                let op = tensor.op; // appease borrowck
                for p_id in op.parents() {
                    self.reuse_parent_or_alloc_node(p_id, &mut ctx)?;
                }

                self.reuse_parent_or_alloc_node(id, &mut ctx)?;
            }

            // Level is processed, time to reuse its nodes' buffers.
            self.free_unused_nodes(&ctx, level);

            trace!(level = i, "finished allocating level");
        }

        Ok(self.max_mem_use)
    }

    fn clear(&mut self) {
        self.free_blocks.clear();
        self.nodes.clear();
        self.max_mem_use = 0;
    }

    fn count_nodes(&mut self, ctx: &Ctx) {
        for id in ctx.graph.nodes().iter().copied() {
            // NB: We only care about the tensors that are going to be allocated by this graph.
            // TODO: we implicitly assume that the tensors belong to this context, we can't verify
            // it here
            let Some(tensor) = ctx.get(id) else { continue };
            if let Some(view) = tensor.view() {
                if ctx.contains(view.origin) {
                    let origin_node = self.nodes.entry(view.origin).or_default();
                    origin_node.n_views += 1;
                }
            }

            for p_id in tensor.op.parents() {
                if ctx.contains(p_id) {
                    let parent_node = self.nodes.entry(p_id).or_default();
                    parent_node.n_children += 1;
                }
            }
        }
    }

    fn reuse_parent_or_alloc_node(&mut self, id: TensorId, ctx: &mut Ctx) -> Result<(), Oom> {
        // NB: Not all nodes are present in the tensors list because the parent of a node may be in
        // another context and thus need not be allocated (e.g. a read-only weight node).
        let Some(tensor) = ctx.get(id) else { return Ok(()) };
        let op = tensor.op;

        if tensor.buf_id() != BufId::unallocated() {
            trace!(tensor = %tensor.name, ?op, "node already allocated");
            return Ok(());
        }

        trace!(tensor = %tensor.name, ?op, "trying to reuse memory");
        if let Some(view) = tensor.view().copied() {
            if let Some(origin) = ctx.get(view.origin) {
                trace!(
                    tensor = %tensor.name,
                    origin = %origin.name,
                    view_offs = view.offs,
                    "reusing view origin memory"
                );
                let buf_id = origin.buf_id;
                let data_offs = origin.data_offs + view.offs;
                ctx[id].reassign_buf(buf_id, data_offs);
                return Ok(());
            }
        } else if op.is_inplace_optimizable() {
            // See if we can reuse a parent's buffer inplace (even if the operation wasn't marked as
            // inplace).
            let mut allocated = false;
            for p_id in op.parents() {
                if allocated {
                    continue;
                }

                let Some(parent) = ctx.get(p_id) else { continue };
                // Disregard nodes not allocated by this allocator.
                // TODO: should this occur actually?
                if parent.buf_id().alloc_id() != self.id.alloc_id() {
                    continue;
                }

                let parent_node = &self.nodes[&p_id];
                if parent_node.has_unique_child() && parent.layout() == ctx[id].layout() {
                    if let Some(parent_view) = parent.view() {
                        // Parent is a view, so we try to reuse its origin buffer.
                        let origin_node = &self.nodes[&parent_view.origin];
                        let origin = &ctx[parent_view.origin];
                        // We only reuse the parent's parent data if they both use the
                        // exact same buffer offset. The offset must be kept so that
                        // the op doesn't overwrite the parent view's data. In theory we could reuse
                        // different portions but then the problem is that the original address of
                        // the allocation is lost, which we need later to free the memory.
                        // We could likely calculate whether the node is allocated within the view
                        // parent's data, but it's error prone so I'm keeping it simpler for now.
                        if origin_node.has_unique_view() && origin.has_same_buf(parent) {
                            trace!(
                                tensor = %ctx[id].name,
                                parent = %parent.name,
                                origin = %origin.name,
                                view_offs = parent_view.offs,
                                "reusing parent view origin memory"
                            );
                            let buf_id = parent.buf_id;
                            let data_offs = parent.data_offs;
                            ctx[id].reassign_buf(buf_id, data_offs);
                            allocated = true;
                        }
                    } else {
                        trace!(
                            tensor = %ctx[id].name,
                            parent = %parent.name,
                            "reusing parent memory"
                        );
                        // We can reuse the parent's buffer.
                        let buf_id = parent.buf_id;
                        let data_offs = parent.data_offs;
                        ctx[id].reassign_buf(buf_id, data_offs);
                        allocated = true;
                    }
                }
            }
            if allocated {
                return Ok(());
            }
        }

        // We couldn't reuse any parent buffer, allocate new memory region for node.
        self.alloc_tensor(&mut ctx[id])
    }

    fn alloc_tensor(&mut self, tensor: &mut Tensor) -> Result<(), Oom> {
        trace!(tensor = %tensor.name, op = ?tensor.op, "allocating new memory");
        let AllocResult { id: buf_id, offs } = self.alloc_node(tensor.size())?;
        tensor.reassign_buf(buf_id, offs);
        Ok(())
    }

    /// Frees nodes in the given level that are not referenced by any downstream node, allowing
    /// downstream nodes to reuse these node's memory.
    ///
    /// We actually reuse a level node's parents rather than the node itself because the current
    /// nodes are likely going to be the parents for the nodes in the next level, whereas the
    /// parents of the current level nodes were likely only used to compute those nodes. That
    /// means we can reuse the parent buffers if they are not referenced by any other node after
    /// accounting for the references from this level.
    fn free_unused_nodes(&mut self, ctx: &Ctx, level: &ConcurLevel) {
        for node_idx in level.iter().copied() {
            let id = ctx.graph.nodes()[node_idx];
            let Some(tensor) = ctx.get(id) else { continue };

            for p_id in tensor.op.parents() {
                let Some(parent) = ctx.get(p_id) else { continue; };
                let parent_node = self.nodes.get_mut(&p_id).unwrap_or_else(|| {
                    // SAFETY: If the parent is present in the tensors list, it is
                    // present in our nodes map.
                    panic!(
                        "Parent '{}' of '{}' not present in nodes map. This is a bug, please report.",
                        parent.name, tensor.name
                    )
                });
                if parent_node.n_children == 0 {
                    // Already freed.
                    continue;
                };
                // Account for this level's node no longer using parent.
                parent_node.n_children -= 1;

                if parent_node.is_unreferenced() {
                    if let Some(view) = parent.view() {
                        let Some(origin) = ctx.get(view.origin) else { continue; };
                        // SAFETY: If the origin node is present in the tensors list, it is
                        // present in our nodes map.
                        let origin_node = self.nodes.get_mut(&view.origin).unwrap();
                        if origin_node.n_views == 0 {
                            // Already freed.
                            continue;
                        };
                        origin_node.n_views -= 1;
                        if origin_node.is_unreferenced() && !origin.has_same_buf(tensor) {
                            self.free_node(origin);
                        }
                    } else if !parent.has_same_buf(tensor) {
                        self.free_node(parent);
                    }
                    // NB: The `has_same_buf` checks work here to check that no
                    // child/view is using the parent data because of the check in
                    // `reuse_parent_or_alloc_node`. See comment there.
                }
            }
        }
    }

    fn alloc_node(&mut self, size: usize) -> Result<AllocResult, Oom> {
        let size = align(size);
        trace!(
            size,
            n_free_blocks = self.free_blocks.len(),
            "allocating new memory"
        );

        if self.free_blocks.is_empty() {
            return Err(Oom {
                from: "graph allocator",
                requested: size,
                free: 0,
                capacity: self.data.len(),
            });
        }

        // Find the best fitting free block besides the last block.
        let mut best_fit_idx = None;
        let mut best_fit_size = usize::MAX;
        let mut max_free = 0; // for debugging
        for (i, block) in self
            .free_blocks
            .iter()
            .take(self.free_blocks.len() - 1)
            .enumerate()
        {
            max_free = max_free.max(block.free);
            if block.free >= size && block.free <= best_fit_size {
                best_fit_idx = Some(i);
                best_fit_size = block.free;
            }
        }

        let best_fit_idx = if let Some(idx) = best_fit_idx {
            idx
        } else {
            let block = self.free_blocks.last().unwrap();
            max_free = max_free.max(block.free);
            if block.free >= size {
                self.free_blocks.len() - 1
            } else {
                return Err(Oom {
                    from: "graph allocator",
                    requested: size,
                    free: max_free,
                    capacity: self.data.len(),
                });
            }
        };

        let block = &mut self.free_blocks[best_fit_idx];
        let tensor_offs = block.offs;
        block.offs += size;
        block.free -= size;
        self.max_mem_use = self.max_mem_use.max(block.offs);

        if block.free == 0 {
            self.free_blocks.remove(best_fit_idx);
        }

        Ok(AllocResult {
            id: self.id,
            offs: tensor_offs,
        })
    }

    fn free_node(&mut self, tensor: &Tensor) {
        debug_assert!(
            tensor.buf_id.is_allocated(),
            "cannot free unallocated tensor '{}' buffer (buf_id={:?})",
            tensor.name,
            tensor.buf_id
        );

        let size = align(tensor.layout.size());
        trace!(
            tensor = %tensor.name,
            size,
            n_free_blocks = self.free_blocks.len(),
            "freeing node memory"
        );

        // See if we can merge with an existing block to avoid fragmentation. We can do this if the
        // tensor is allocated either at the start or at the end of the block.
        for i in 0..self.free_blocks.len() {
            // Check if the tensor is allocated at the start of this block.
            let block = &mut self.free_blocks[i];
            if tensor.data_offs + size == block.offs {
                block.offs = tensor.data_offs;
                block.free += size;

                let FreeBlock { offs, free } = *block;
                // Try to merge with the previous block.
                if i > 0 {
                    let prev_block = &mut self.free_blocks[i - 1];
                    if prev_block.end_offs() == offs {
                        prev_block.free += free;
                        self.free_blocks.remove(i);
                    }
                }
                return;
            }

            // Check if the tensor is allocated at the end of this block.
            if block.offs <= usize::MAX - block.free && block.end_offs() == tensor.data_offs
            // TODO: should this be block.free == 0 && block.offs == tensor.data_offs + size?
            {
                // The tensor is in this block.
                block.free += size;
                // Try to merge with the block if the tensor is reaching exactly till the end of
                // this block (and so the start of the next one).
                if i < self.free_blocks.len() - 1 {
                    let FreeBlock {
                        offs: next_offs,
                        free: next_free,
                    } = self.free_blocks[i + 1];
                    let block = &mut self.free_blocks[i]; // reborrow for borrowck
                    if block.end_offs() == next_offs {
                        block.free += next_free;
                        self.free_blocks.remove(i + 1);
                    }
                }
                return;
            }
        }

        // Insert the new free block in the correct position to keep the array sorted by address (to
        // make merging blocks faster).
        let mut insert_pos = 0;
        for block in &self.free_blocks {
            if block.offs >= tensor.data_offs {
                break;
            }
            insert_pos += 1;
        }
        trace!(
            insert_pos,
            before_block = ?self.free_blocks.get(insert_pos),
            n_free_blocks = self.free_blocks.len(),
            "inserting new block"
        );
        self.free_blocks.insert(
            insert_pos,
            FreeBlock {
                offs: tensor.data_offs,
                free: size,
            },
        );
    }
}

/// Helper type to make it easier to pass around values and provides safe getters.
struct Ctx<'a> {
    graph: &'a ComputeGraph,
    tensors: &'a mut [Tensor],
    ctx_id: CtxId,
}

impl Ctx<'_> {
    fn get(&self, id: TensorId) -> Option<&Tensor> {
        if id.ctx() == self.ctx_id {
            Some(&self.tensors[id.idx()])
        } else {
            None
        }
    }

    fn contains(&self, id: TensorId) -> bool {
        id.ctx() == self.ctx_id && id.idx() < self.tensors.len()
    }
}

impl Index<TensorId> for Ctx<'_> {
    type Output = Tensor;
    fn index(&self, id: TensorId) -> &Self::Output {
        debug_assert_eq!(id.ctx(), self.ctx_id);
        &self.tensors[id.idx()]
    }
}
impl IndexMut<TensorId> for Ctx<'_> {
    fn index_mut(&mut self, id: TensorId) -> &mut Self::Output {
        debug_assert_eq!(id.ctx(), self.ctx_id);
        &mut self.tensors[id.idx()]
    }
}

/// The alignment is 4 bytes because the largest tensor type is f32.
const ALIGNMENT: usize = 32;

fn align_buf(buf: &mut [u8]) -> &mut [u8] {
    buf::align_buf(buf, ALIGNMENT)
}

fn align(t: usize) -> usize {
    buf::align(t, ALIGNMENT)
}

pub struct GraphArgs {
    pub size: usize,
    pub prealloc: bool,
}

impl<B: Deref<Target = [u8]>> TensorAlloc for GraphAlloc<B> {
    type Arg = GraphArgs;

    fn alloc(&mut self, args: Self::Arg) -> Result<AllocResult, Oom> {
        if args.prealloc {
            self.alloc_node(args.size)
        } else {
            Ok(AllocResult {
                id: BufId::unallocated(),
                offs: 0,
            })
        }
    }
}

impl<B: DerefMut<Target = [u8]>> ReadWriteAlloc for GraphAlloc<B> {
    #[inline(always)]
    fn get_slice(&self, id: BufId, offs: usize, len: usize) -> Option<&[u8]> {
        if id == self.id && self.data.len() >= offs + len {
            Some(&self.data[offs..offs + len])
        } else {
            None
        }
    }

    #[inline(always)]
    fn get_mut_slice(&mut self, id: BufId, offs: usize, len: usize) -> Option<&mut [u8]> {
        if id == self.id && self.data.len() >= offs + len {
            Some(&mut self.data[offs..offs + len])
        } else {
            None
        }
    }
}

impl<B> AllocSize for GraphAlloc<B> {
    #[inline(always)]
    fn alloc_size(&self) -> usize {
        self.max_mem_use
    }
}

impl Op {
    #[inline(always)]
    fn is_inplace_optimizable(&self) -> bool {
        match self {
            Op::Unary { .. } => true,
            Op::Binary { op, .. } => matches!(
                op,
                BinaryOp::Add | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Copy
            ),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::{NoSrc, TensorOptionsExt},
        op::{Noop, UnaryOp},
        tensor::{MemoryLocation, View},
        test_utils::init_tracing,
        Layout, TensorBuilder, D1, D2,
    };

    use super::*;

    #[test]
    fn allocs_single_op_graph() {
        init_tracing();

        let mut alloc = GraphAlloc::measure();
        let mut tb = TensorBuilder::graph_alloc_with_src(&mut alloc, NoSrc);
        let mem_limit = usize::MAX;

        let dims = D2 { rows: 2, cols: 5 };
        let a = tb.new_tensor(dims.with_name("a")).unwrap();
        let b = tb.new_tensor(dims.with_name("b")).unwrap();

        let ab = tb.add(a, b).unwrap();
        tb[ab].set_name("a+b");

        let mut graph = ComputeGraph::default();
        graph.expand_forward(ab, &tb).unwrap();
        graph.find_exec_plan(mem_limit, &tb).unwrap();

        let alloc_size = tb.alloc_graph(&graph, mem_limit).unwrap();

        println!("tensor_size={} alloc_size={alloc_size}", tb[a].size());

        assert_eq!(alloc_size, align(tb[a].size()) * 2);
    }

    #[test]
    fn allocs_simple_graph() {
        init_tracing();

        let mut alloc = GraphAlloc::measure();
        let mut tb = TensorBuilder::graph_alloc_with_src(&mut alloc, NoSrc);
        let mem_limit = usize::MAX;

        let dims = D2 { rows: 2, cols: 5 };
        let a = tb.new_tensor(dims.with_name("a")).unwrap();
        let b = tb.new_tensor(dims.with_name("b")).unwrap();
        let c = tb.new_tensor(dims.with_name("c")).unwrap();
        let d = tb.new_tensor(dims.with_name("d")).unwrap();

        let ab = tb.add(a, b).unwrap();
        tb[ab].set_name("a+b");
        let abc = tb.add(ab, c).unwrap();
        tb[abc].set_name("ab+c");
        let abcd = tb.add(abc, d).unwrap();
        tb[abcd].set_name("abc+d");

        let mut graph = ComputeGraph::default();
        graph.expand_forward(abcd, &tb).unwrap();
        graph.find_exec_plan(mem_limit, &tb).unwrap();

        let alloc_size = tb.alloc_graph(&graph, mem_limit).unwrap();

        println!("tensor_size={} alloc_size={alloc_size}", tb[a].size());

        // This is the most optimal case: the allocation size is simply the 4 identically sized
        // input tensors and the operations reuse these tensors.
        // One of the initial input nodes gets deallocated, however this is typically not how a real
        // world scenario would look because the input nodes are usually weights.
        assert_eq!(alloc_size, align(tb[a].size()) * 3);
    }

    #[test]
    fn allocs_concur_graph() {
        init_tracing();

        let mut alloc = GraphAlloc::measure();
        let mut tb = TensorBuilder::graph_alloc_with_src(&mut alloc, NoSrc);
        let mem_limit = usize::MAX;

        let dims = D2 { rows: 2, cols: 5 };
        let a0 = tb.new_tensor(dims.with_name("a0")).unwrap();
        let b0 = tb.new_tensor(dims.with_name("b0")).unwrap();
        let a1 = tb.new_tensor(dims.with_name("a1")).unwrap();
        let b1 = tb.new_tensor(dims.with_name("b1")).unwrap();

        let ab0 = tb.add(a0, b0).unwrap();
        tb[ab0].set_name("a0+b0");
        let ab1 = tb.add(a1, b1).unwrap();
        tb[ab1].set_name("a1+b1");

        let output = tb.add(ab0, ab1).unwrap();
        tb[output].set_name("output");

        let mut graph = ComputeGraph::default();
        graph.expand_forward(output, &tb).unwrap();
        graph.find_exec_plan(mem_limit, &tb).unwrap();

        let alloc_size = tb.alloc_graph(&graph, mem_limit).unwrap();

        println!("tensor_size={} alloc_size={alloc_size}", tb[a0].size());

        /// Similar to the above, all operations are able to use their inputs. Although both
        /// b input tensors get deallocated, they are at the start of the graph so the allocation
        /// size does not change.
        assert_eq!(alloc_size, align(tb[a0].size()) * 4);
    }

    #[test]
    fn cannot_reuse_shared_parent_node_mem() {
        let mut alloc = GraphAlloc::measure();

        let _pad = new_tensor(0, D1 { cols: 1024 });
        let dims = D2 {
            rows: 32002,
            cols: 8,
        };
        let mut parent = new_tensor(0, dims);
        let mut child0 = new_tensor(1, dims);
        child0.op = Op::Unary {
            op: UnaryOp::Noop(Noop::Permute),
            a: parent.id(),
        };
        let mut child1 = new_tensor(2, dims);
        child1.op = Op::Unary {
            op: UnaryOp::Noop(Noop::Permute),
            a: parent.id(),
        };

        alloc.nodes.insert(
            parent.id(),
            Node {
                n_children: 2,
                n_views: 0,
            },
        );

        alloc.alloc_tensor(&mut parent).unwrap();

        let child0_id = child0.id();
        let child1_id = child1.id();
        let graph = ComputeGraph::default();
        let mut tensors = [parent, child0, child1];
        let mut ctx = Ctx {
            graph: &graph,
            tensors: &mut tensors,
            ctx_id: 0,
        };
        alloc
            .reuse_parent_or_alloc_node(child0_id, &mut ctx)
            .unwrap();
        alloc
            .reuse_parent_or_alloc_node(child1_id, &mut ctx)
            .unwrap();

        dbg!(&tensors);

        assert_eq!(tensors[0].buf_id, tensors[1].buf_id);
        assert_ne!(tensors[0].data_offs, tensors[1].data_offs);

        assert_eq!(tensors[0].buf_id, tensors[2].buf_id);
        assert_ne!(tensors[0].data_offs, tensors[2].data_offs);
    }

    #[test]
    fn reuses_parent_node_mem() {
        let mut alloc = GraphAlloc::measure();

        let _pad = new_tensor(0, D1 { cols: 1024 });
        let dims = D2 {
            rows: 32002,
            cols: 8,
        };
        let mut parent = new_tensor(0, dims);
        let mut child = new_tensor(1, dims);
        child.op = Op::Unary {
            op: UnaryOp::Noop(Noop::Permute),
            a: parent.id(),
        };

        alloc.nodes.insert(
            parent.id(),
            Node {
                n_children: 1,
                n_views: 0,
            },
        );

        alloc.alloc_tensor(&mut parent).unwrap();

        let child_id = child.id();
        let graph = ComputeGraph::default();
        let mut tensors = [parent, child];
        let mut ctx = Ctx {
            graph: &graph,
            tensors: &mut tensors,
            ctx_id: 0,
        };
        alloc
            .reuse_parent_or_alloc_node(child_id, &mut ctx)
            .unwrap();

        dbg!(&tensors);

        assert_eq!(tensors[1].buf_id, tensors[0].buf_id);
        assert_eq!(tensors[1].data_offs, tensors[0].data_offs);
    }

    #[test]
    fn reuses_view_origin_mem() {
        let mut alloc = GraphAlloc::measure();

        let _pad = new_tensor(0, D1 { cols: 1024 });
        let dims = D2 {
            rows: 32002,
            cols: 8,
        };
        let mut parent = new_tensor(0, dims);
        let mut child = new_tensor(1, dims);
        child.view = Some(View {
            origin: parent.id(),
            offs: 0,
        });

        alloc.nodes.insert(
            parent.id(),
            Node {
                n_children: 0,
                n_views: 1,
            },
        );

        alloc.alloc_tensor(&mut parent).unwrap();

        let child_id = child.id();
        let graph = ComputeGraph::default();
        let mut tensors = [parent, child];
        let mut ctx = Ctx {
            graph: &graph,
            tensors: &mut tensors,
            ctx_id: 0,
        };
        alloc
            .reuse_parent_or_alloc_node(child_id, &mut ctx)
            .unwrap();

        dbg!(&tensors);

        assert_eq!(tensors[1].buf_id, tensors[0].buf_id);
        assert_eq!(tensors[1].data_offs, tensors[0].data_offs);
    }

    #[test]
    fn reuses_parent_view_origin_mem() {
        let mut alloc = GraphAlloc::measure();

        let _pad = new_tensor(0, D1 { cols: 1024 });
        let dims = D2 {
            rows: 32002,
            cols: 8,
        };
        let mut grandparent = new_tensor(0, dims);
        let mut parent = new_tensor(1, dims);
        parent.view = Some(View {
            origin: grandparent.id(),
            offs: 0,
        });
        let mut child = new_tensor(2, dims);
        child.op = Op::Unary {
            op: UnaryOp::Noop(Noop::Permute),
            a: parent.id(),
        };

        alloc.nodes.insert(
            grandparent.id(),
            Node {
                n_children: 0,
                n_views: 1,
            },
        );
        alloc.nodes.insert(
            parent.id(),
            Node {
                n_children: 1,
                n_views: 0,
            },
        );

        alloc.alloc_tensor(&mut grandparent).unwrap();

        let child_id = child.id();
        let parent_id = parent.id();
        let graph = ComputeGraph::default();
        let mut tensors = [grandparent, parent, child];
        let mut ctx = Ctx {
            graph: &graph,
            tensors: &mut tensors,
            ctx_id: 0,
        };
        alloc
            .reuse_parent_or_alloc_node(parent_id, &mut ctx)
            .unwrap();

        alloc
            .reuse_parent_or_alloc_node(child_id, &mut ctx)
            .unwrap();

        dbg!(&tensors);

        assert_eq!(tensors[1].buf_id, tensors[0].buf_id);
        assert_eq!(tensors[1].data_offs, tensors[0].data_offs);
        assert_eq!(tensors[2].buf_id, tensors[0].buf_id);
        assert_eq!(tensors[2].data_offs, tensors[0].data_offs);
    }

    #[test]
    fn allocs_and_frees_nodes() {
        let mut alloc = GraphAlloc::measure();

        let mut node0 = new_tensor(
            0,
            D2 {
                rows: 32002,
                cols: 8,
            },
        );
        let mut node1 = new_tensor(
            1,
            D2 {
                rows: 1024,
                cols: 1024,
            },
        );
        let mut node2 = new_tensor(
            2,
            D2 {
                rows: 1024,
                cols: 256,
            },
        );

        println!("allocating node0: size={}", node0.size());
        alloc.alloc_tensor(&mut node0).unwrap();
        println!("allocating node1: size={}", node1.size());
        alloc.alloc_tensor(&mut node1).unwrap();
        println!("allocating node2: size={}", node2.size());
        alloc.alloc_tensor(&mut node2).unwrap();

        println!("{:?}", alloc.free_blocks);

        // This should create a split in the free block.
        println!("freeing node1");
        alloc.free_node(&node1);

        println!("{:?}", alloc.free_blocks);
        assert_eq!(alloc.free_blocks.len(), 2);
        assert_eq!(alloc.free_blocks[0].offs, node1.data_offs());
        assert_eq!(alloc.free_blocks[0].free, node1.size());
        assert_eq!(
            alloc.free_blocks[1].offs,
            align(node0.size()) + align(node1.size()) + align(node2.size())
        );
    }

    fn new_tensor(idx: u32, dims: impl Into<Layout>) -> Tensor {
        Tensor {
            id: TensorId::new(0, idx),
            layout: dims.into(),
            op: Op::Const,
            data_offs: 0,
            buf_id: BufId::unallocated(),
            name: "".into(),
            view: None,
            loc: MemoryLocation::default(),
        }
    }
}
