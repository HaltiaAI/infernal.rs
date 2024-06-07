use core::fmt;
use std::{borrow::Cow, ops::Deref};

use crate::{
    alloc::{Oom, TensorAlloc},
    builder::{AutoOpName, TensorBuilder},
    layout::{BATCH, D2, D3, DEPTH, MAX_DIMS, ROW},
    tensor::View,
    Dims, Layout, Tensor, TensorId, TensorSrc, Type, D1, D4,
};

pub use Mode::*;

use Type::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    Inplace,
    Dup,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "graph-inspect",
    derive(strum::EnumDiscriminants),
    strum_discriminants(derive(serde::Serialize, serde::Deserialize))
)]
pub enum Op {
    /// The node is a constant value. Usually an input to a computation.
    Const,
    Unary {
        op: UnaryOp,
        a: TensorId,
    },
    Binary {
        op: BinaryOp,
        a: TensorId,
        b: TensorId,
    },
}

impl Op {
    pub fn parents(&self) -> impl IntoIterator<Item = TensorId> {
        let mut p = [None; 2];
        match self {
            Self::Unary { a, .. } => {
                p[0] = Some(*a);
            }
            Self::Binary { a, b, .. } => {
                p[0] = Some(*a);
                p[1] = Some(*b);
            }
            _ => (),
        }
        p.into_iter().flatten()
    }

    #[inline(always)]
    #[deprecated(note = "use `parents` returning an iterator instead")]
    pub fn try_visit_parents<E>(
        &self,
        mut f: impl FnMut(TensorId) -> Result<(), E>,
    ) -> Result<(), E> {
        match self {
            Op::Unary { a, .. } => f(*a)?,
            Op::Binary { a, b, .. } => {
                f(*a)?;
                f(*b)?;
            }
            _ => (),
        }
        Ok(())
    }

    #[inline(always)]
    #[deprecated(note = "use `parents` returning an iterator instead")]
    pub fn visit_parents(&self, mut f: impl FnMut(TensorId)) {
        match self {
            Op::Unary { a, .. } => f(*a),
            Op::Binary { a, b, .. } => {
                f(*a);
                f(*b);
            }
            _ => (),
        }
    }

    pub fn is_noop(&self) -> bool {
        match self {
            Op::Const => true,
            Op::Unary { op, .. } => matches!(op, UnaryOp::Noop(_)),
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RopeExpanded {
    pub n_past: i32,
    pub n_dims: i32,
    pub n_ctx: i32,
    pub mode: i32,
    // pub n_orig_ctx: usize,
    pub custom: RopeCustom,
    // pub ext_factor: f32,
    // pub attn_factor: f32,
    // pub beta_fast: f32,
    // pub beta_slow: f32,
    // TODO: These were added later to ggml.
}

impl From<Rope> for RopeExpanded {
    fn from(value: Rope) -> Self {
        Self {
            // kernels expect i32, simpler to do conversion here
            n_past: value.n_past as i32,
            n_dims: value.n_dims as i32,
            n_ctx: value.n_ctx as i32,
            mode: value.mode as i32,
            // n_orig_ctx: value.n_orig_ctx,
            custom: value.custom.unwrap_or_default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Rope {
    pub n_past: usize,
    pub n_dims: usize,
    pub n_ctx: usize,
    pub mode: usize,
    // pub n_orig_ctx: usize,
    pub custom: Option<RopeCustom>,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RopeCustom {
    pub freq_base: f32,
    pub freq_scale: f32,
    // pub xpos_base: f32,
    // pub ext_factor: f32,
    // pub attn_factor: f32,
    // pub beta_fast: f32,
    // pub beta_slow: f32,
    // TODO: These were added later to ggml.
}

impl Default for RopeCustom {
    #[inline(always)]
    fn default() -> Self {
        Self {
            freq_base: 10_000.,
            freq_scale: 1.,
            // xpos_base: 0.,
        }
    }
}

// impl Op {
//     #[inline(always)]
//     pub fn sources<const N: usize>(&self) -> [TensorId; N] {
//         // this is hacky but meh
//         match self {
//             Op::Const => {
//                 assert_eq!(N, 0);
//                 [TensorId::new(0, 0); N]
//             }
//             Op::Unary { x, .. } => {
//                 assert_eq!(N, 1);
//                 [*x; N]
//             }
//             Op::Binary { x, y, .. } => {
//                 assert_eq!(N, 2);
//                 let mut res = [TensorId::new(0, 0); N];
//                 res[0] = *x;
//                 res[1] = *y;
//                 res
//             }
//         }
//     }
// }

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Op::*;
        match self {
            Const => write!(f, "const"),
            Unary { op, a: x } => write!(f, "{op}({x})"),
            Binary { op, a: x, b: y } => write!(f, "({x} {op} {y})"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "graph-inspect",
    derive(strum::EnumDiscriminants),
    strum_discriminants(derive(serde::Serialize, serde::Deserialize, strum::Display))
)]
pub enum UnaryOp {
    /// The node is not a constant value but does not need explicit execution.
    ///
    /// Operations that are noops:
    /// - reshape
    /// - transpose
    /// - view
    /// - permute
    ///
    /// Since noops are not executed the op variant is only used for debugging.
    Noop(Noop),
    RmsNorm {
        eps: f32,
    },
    Rope(RopeExpanded),
    DiagMaskInf {
        n_past: usize,
    },
    Softmax {
        scale: f32,
        mask: Option<TensorId>,
    },
    Silu,
    Relu,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "graph-inspect",
    derive(strum::EnumDiscriminants),
    strum_discriminants(derive(serde::Serialize, serde::Deserialize, strum::Display))
)]
pub enum Noop {
    Reshape,
    Transpose,
    View,
    Permute,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "graph-inspect",
    derive(strum::EnumDiscriminants),
    strum_discriminants(derive(serde::Serialize, serde::Deserialize, strum::Display))
)]
pub enum BinaryOp {
    Matmul,
    Add,
    Mul,
    Div,
    /// The x tensor is the one that is scaled, the y tensor is the scalar.
    Scale,
    Im2Col(Im2Col),
    GetRows,
    Copy,
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Im2Col {
    pub stride0: usize,
    pub stride1: usize,
    pub padding0: usize,
    pub padding1: usize,
    pub dilation0: usize,
    pub dilation1: usize,
}

impl fmt::Display for Noop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Noop::*;
        match self {
            Reshape => write!(f, "reshape"),
            Transpose => write!(f, "transpose"),
            View => write!(f, "view"),
            Permute => write!(f, "permute"),
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use UnaryOp::*;
        match self {
            Noop(op) => write!(f, "{}", op),
            RmsNorm { eps } => write!(f, "rms_norm[ε={eps}]"),
            Rope(args) => write!(f, "rope_custom[{args:?}]"),
            DiagMaskInf { n_past } => write!(f, "diag_mask_inf[n={n_past}]"),
            Softmax { scale, .. } => write!(f, "softmax[{scale}]"),
            Silu => write!(f, "silu"),
            Relu => write!(f, "relu"),
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use BinaryOp::*;
        match self {
            Matmul => write!(f, "⋅"),
            Add => write!(f, "+"),
            Mul => write!(f, "*"),
            Div => write!(f, "/"),
            Scale => write!(f, "scale_by"),
            // TODO: this is prob quite unreadable
            Im2Col(v) => write!(
                f,
                "im2col[s0={stride0}, s1={stride1}, p0={padding0}, p1={padding1}, d0={dilation0}, d1={dilation1}]",
                stride0 = v.stride0,
                stride1 = v.stride1,
                padding0 = v.padding0,
                padding1 = v.padding1,
                dilation0 = v.dilation0,
                dilation1 = v.dilation1,
            ),
            GetRows => write!(f, "get_rows"),
            Copy => write!(f, "copy ->"),
        }
    }
}

/// Returns the name of the operation function the macro is invoked in.
macro_rules! __op_func__ {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let path = type_name_of(f);
        // get rid of the ::f suffix
        let path = &path[..path.len() - 3];
        // only use last path element to get our fn name
        let name = path.rsplit("::").next().unwrap();
        // the fn may be inside an impl helper, remove as it's confisuing for API user
        name.trim_end_matches("_impl")
    }};
}

macro_rules! op_assert {
    ($cond:expr) => {{
        assert!($cond, "{}: {}", __op_func__!(), stringify!($cond))
    }};
    ($cond:expr, $($arg:tt)+) => {{
        assert!($cond, "{}: {}: {}", __op_func__!(), stringify!($cond), format_args!($($arg)+))
    }};
}

macro_rules! op_assert_eq {
    ($lhs:expr, $rhs:expr) => {{
        assert_eq!($lhs, $rhs, "{}", __op_func__!())
    }};
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {{
        assert_eq!($lhs, $rhs, "{}: {}", __op_func__!(), format_args!($($arg)+))
    }};
}

macro_rules! op_assert_ne {
    ($lhs:expr, $rhs:expr) => {{
        assert_ne!($lhs, $rhs, "{}", __op_func__!())
    }};
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {{
        assert_ne!($lhs, $rhs, "{}: {}", __op_func__!(), format_args!($($arg)+))
    }};
}

// ---- ops ----
impl<A: TensorAlloc, S: TensorSrc> TensorBuilder<A, S>
where
    // Layout: ToAllocArg<A>,
    A::Arg: for<'a> From<&'a Layout>,
{
    pub fn transpose(&mut self, a_id: TensorId) -> TensorId {
        let res_id = self
            .new_op_based_on_tensor(
                a_id,
                Op::Unary {
                    op: UnaryOp::Noop(Noop::Transpose),
                    a: a_id,
                },
                Inplace,
            )
            .expect(INPLACE_EXPECT);

        let res = &mut self[res_id];

        let mut dims = res.dims;
        dims[0] = res.dims[1];
        dims[1] = res.dims[0];

        let mut stride = res.stride;
        stride[0] = res.stride[1];
        stride[1] = res.stride[0];

        res.layout = Layout::from_raw(res.n(), dims, stride, res.typ());

        res_id
    }

    /// Performs a matrix-matrix multiplication.
    ///
    /// The rhs of the multiplication is transposed before the operation, therefore, unlike with
    /// normal matmuls where `X = M x K` and `Y = K x N`, here the operands have to have their
    /// columns matched up:  `X = M x K` and `Y = N x K`. The result, too, is transposed
    /// into `N x M`.
    ///
    /// Note broadcasting is not supported to avoid common bugs related to it.
    pub fn matmul(&mut self, a_id: TensorId, b_id: TensorId) -> Result<TensorId, Oom> {
        let a = &self[a_id];
        let b = &self[b_id];
        op_assert_eq!(
            a.cols(),
            b.cols(),
            "expected format: M x N * K x N (rhs dimensions reversed), \
            that is, columns must match, but they are: '{a}'={a_cols} '{b}'={b_cols}",
            a = a.name,
            b = b.name,
            a_cols = a.cols(),
            b_cols = b.cols(),
        );
        // TODO:
        // assert_eq!(a.n(), b.n(), "matmul tensors must be of the same dimension");
        // TODO figure out why this constraint exists -- likely related to the kernel?
        op_assert!(
            !b.is_transposed(),
            "b tensor '{b}' must not be already transposed",
            b = b.name
        );
        match (a.typ(), b.typ()) {
            (F32, F32) => (),
            (F32, F16) => (),
            (F16, F16) => (),
            (F16, F32) => (),
            (Q2_K | Q3_K | Q4_K | Q5_K | Q6_K, F32) => (),
            _ => panic!(
                "unsupported type combination in matmul: {a}={a_typ:?} x {b}={b_typ:?}",
                a = a.name,
                b = b.name,
                a_typ = a.typ(),
                b_typ = b.typ()
            ),
        }

        // A: k columns, n rows => [ne03, ne02, n, k]
        // B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
        // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]

        // y is transposed before the matmul operator, so we take its rows instead of columns as the
        // second dimension.
        // TODO: find out why this is and whether we can change this in the kernel
        // TODO: document y <> x order switchup. looks like ggml transposes the result matrix too
        let mut layout = Layout::new_raw_dims(&[a.rows(), b.rows(), b.depth(), b.batches()], F32);
        layout.n = a.n.max(b.n);
        // let buf = self.alloc.alloc(ToAllocArg::<A>::to_alloc_arg(&layout))?;
        let buf = self.alloc.alloc(A::Arg::from(&layout))?;
        let op = Op::Binary {
            op: BinaryOp::Matmul,
            a: a_id,
            b: b_id,
        };
        Ok(self.new_tensor_impl(
            layout,
            op,
            self.get_op_name(op),
            buf.id,
            buf.offs,
            None,
            None,
        ))
    }

    /// Performs matrix addition.
    ///
    /// The returned tensor is a copy of the input tensor.
    #[inline(always)]
    pub fn add(&mut self, a_id: TensorId, b_id: TensorId) -> Result<TensorId, Oom> {
        self.add_impl(a_id, b_id, Dup)
    }

    /// Performs matrix addition.
    ///
    /// The returned tensor is a view of the input tensor and therefore its underlying buffer is
    /// modified.
    #[inline(always)]
    pub fn add_inplace(&mut self, a_id: TensorId, b_id: TensorId) -> TensorId {
        self.add_impl(a_id, b_id, Inplace).expect(INPLACE_EXPECT)
    }

    fn add_impl(&mut self, a_id: TensorId, b_id: TensorId, mode: Mode) -> Result<TensorId, Oom> {
        let a = &self[a_id];
        let b = &self[b_id];
        // TODO: is this right? look at the metal exec op
        op_assert_eq!(a.rows(), b.rows());
        op_assert_eq!(a.cols(), b.cols());
        // TODO: support other types
        op_assert_eq!(a.typ(), b.typ());

        self.new_op_based_on_tensor(
            a_id,
            Op::Binary {
                op: BinaryOp::Add,
                a: a_id,
                b: b_id,
            },
            mode,
        )
    }

    /// Performs matrix-vector (TODO or scalar?) multiplication.
    ///
    /// The returned tensor is a copy of the input tensor.
    #[inline(always)]
    pub fn mul(&mut self, a_id: TensorId, b_id: TensorId) -> Result<TensorId, Oom> {
        self.mul_impl(a_id, b_id, Dup)
    }

    /// Performs matrix-vector (TODO or scalar?) multiplication.
    ///
    /// The returned tensor is a view of the input tensor and therefore its underlying buffer is
    /// modified.
    #[inline(always)]
    pub fn mul_inplace(&mut self, a_id: TensorId, b_id: TensorId) -> TensorId {
        // TODO: more elegant way would be to make fn generic over return type and pass in a closure
        // for producting the buffer id. mode could instead be a taged type that has a trat impl for
        // this. not prio.
        self.mul_impl(a_id, b_id, Inplace).expect(INPLACE_EXPECT)
    }

    fn mul_impl(&mut self, a_id: TensorId, b_id: TensorId, mode: Mode) -> Result<TensorId, Oom> {
        let a = &self[a_id];
        let b = &self[b_id];
        op_assert!(
            can_repeat(b, a),
            "b tensor '{b}' ({b_layout}) must be repeatable in terms of a tensor '{a}' ({a_layout})",
            a = a.name,
            b = b.name,
            a_layout = a.layout,
            b_layout = b.layout,
        );

        self.new_op_based_on_tensor(
            a_id,
            Op::Binary {
                op: BinaryOp::Mul,
                a: a_id,
                b: b_id,
            },
            mode,
        )
    }

    #[inline(always)]
    pub fn rms_norm(&mut self, a_id: TensorId, eps: f32) -> Result<TensorId, Oom> {
        self.rms_norm_impl(a_id, eps, Dup)
    }

    #[inline(always)]
    pub fn rms_norm_inplace(&mut self, a_id: TensorId, eps: f32) -> TensorId {
        self.rms_norm_impl(a_id, eps, Inplace)
            .expect(INPLACE_EXPECT)
    }

    #[inline(always)]
    fn rms_norm_impl(&mut self, a_id: TensorId, eps: f32, mode: Mode) -> Result<TensorId, Oom> {
        self.new_op_based_on_tensor(
            a_id,
            Op::Unary {
                a: a_id,
                op: UnaryOp::RmsNorm { eps },
            },
            mode,
        )
    }

    #[inline(always)]
    pub fn rope(&mut self, a_id: TensorId, args: Rope) -> Result<TensorId, Oom> {
        self.rope_impl(a_id, args, Dup)
    }

    #[inline(always)]
    pub fn rope_inplace(&mut self, a_id: TensorId, args: Rope) -> TensorId {
        self.rope_impl(a_id, args, Inplace).expect(INPLACE_EXPECT)
    }

    #[inline(always)]
    fn rope_impl(&mut self, a_id: TensorId, args: Rope, mode: Mode) -> Result<TensorId, Oom> {
        self.new_op_based_on_tensor(
            a_id,
            Op::Unary {
                a: a_id,
                op: UnaryOp::Rope(args.into()),
            },
            mode,
        )
    }

    /// Copies data from one tensor to another.
    ///
    /// Note that ggml has a few aliases to copy (cpy): cont, dup. We're not implementing those,
    /// just use this.
    #[inline(always)]
    pub fn copy(&mut self, from_id: TensorId, to_id: TensorId) -> TensorId {
        let from = &self[from_id];
        let to = &self[to_id];
        op_assert_eq!(
            from.total_elems(),
            to.total_elems(),
            "src '{from}' and dst '{to}' total elements must match",
            from = from.name,
            to = to.name
        );
        match from.typ() {
            F32 => op_assert!(
                to.typ() == F32 || to.typ() == F16,
                "can only copy to f32 or f16 from f32"
            ),
            F16 => op_assert!(to.typ() == F16, "can only copy to f16 from f16"),
            _ => panic!("can only copy f32 or f16 tensors"),
        }

        let name = format!("{} (copy of {})", to.name, from.name);
        let res = self
            .new_op_based_on_tensor(
                to_id,
                Op::Binary {
                    op: BinaryOp::Copy,
                    a: from_id,
                    b: to_id,
                },
                Inplace,
            )
            .expect(INPLACE_EXPECT);
        // a bit hacky but whatever
        self.tensors[res.idx()].name = name.into();
        res
    }

    pub fn permute(&mut self, a_id: TensorId, axes: [usize; MAX_DIMS]) -> TensorId {
        op_assert!(axes[0] < MAX_DIMS);
        op_assert!(axes[1] < MAX_DIMS);
        op_assert!(axes[2] < MAX_DIMS);
        op_assert!(axes[3] < MAX_DIMS);

        op_assert_ne!(axes[0], axes[1]);
        op_assert_ne!(axes[0], axes[2]);
        op_assert_ne!(axes[0], axes[3]);
        op_assert_ne!(axes[1], axes[2]);
        op_assert_ne!(axes[1], axes[3]);
        op_assert_ne!(axes[2], axes[3]);

        let a = &self[a_id];

        let mut dims = a.dims;
        dims[axes[0]] = a.dims[0];
        dims[axes[1]] = a.dims[1];
        dims[axes[2]] = a.dims[2];
        dims[axes[3]] = a.dims[3];

        let mut stride = a.stride;
        stride[axes[0]] = a.stride[0];
        stride[axes[1]] = a.stride[1];
        stride[axes[2]] = a.stride[2];
        stride[axes[3]] = a.stride[3];

        let layout = Layout::from_raw(a.n(), dims, stride, a.typ());
        let op = Op::Unary {
            op: UnaryOp::Noop(Noop::Permute),
            a: a_id,
        };

        let base_id = if let Some(parent_view) = a.view() {
            parent_view.origin
        } else {
            a_id
        };
        self.new_tensor_impl(
            layout,
            op,
            self.get_op_name(op),
            a.buf_id,
            a.data_offs,
            Some(View {
                origin: base_id,
                offs: 0,
            }),
            None,
        )
    }

    pub fn reshape(&mut self, a_id: TensorId, dims: impl Into<Dims>) -> TensorId {
        let a = &self[a_id];
        let cur_elems = a.total_elems();
        let dims = dims.into();
        let reshaped_elems = dims.total_elems();
        op_assert_eq!(
            cur_elems,
            reshaped_elems,
            "tensor '{a}' size must be preserved: current={cur_elems} reshaped={reshaped_elems}",
            a = a.name,
        );
        // TODO: check if tensor is contiguous

        let layout = Layout::new_raw_dims(dims.as_slice(), a.typ());
        let op = Op::Unary {
            op: UnaryOp::Noop(Noop::Reshape),
            a: a_id,
        };
        let base_id = if let Some(parent_view) = a.view() {
            parent_view.origin
        } else {
            a_id
        };
        self.new_tensor_impl(
            layout,
            op,
            Some(format!("{} (reshaped)", a.name).into()),
            a.buf_id,
            a.data_offs,
            Some(View {
                origin: base_id,
                offs: 0,
            }),
            None,
        )
    }

    pub fn view_1d(&mut self, a_id: TensorId, dims: impl Into<D1>, byte_offs: usize) -> TensorId {
        self.view_impl(a_id, dims.into().into(), byte_offs)
    }

    /// Note that the stride columns are not used and can be set to a default value.
    pub fn view_2d(&mut self, a_id: TensorId, dims: D2, stride: D2, byte_offs: usize) -> TensorId {
        let res_id = self.view_impl(a_id, dims.into(), byte_offs);

        let res = &mut self[res_id];
        res.layout.stride[ROW] = stride.rows;
        res.layout.stride[DEPTH] = res.stride[ROW] * dims.rows;
        res.layout.stride[BATCH] = res.stride[DEPTH];

        res_id
    }

    /// Note that the stride columns are not used and can be set to a default value.
    pub fn view_3d(&mut self, a_id: TensorId, dims: D3, stride: D3, byte_offs: usize) -> TensorId {
        let res_id = self.view_impl(a_id, dims.into(), byte_offs);

        let res = &mut self[res_id];
        res.layout.stride[ROW] = stride.rows;
        res.layout.stride[DEPTH] = stride.depth;
        res.layout.stride[BATCH] = res.stride[DEPTH] * dims.depth;

        res_id
    }

    fn view_impl(&mut self, src_id: TensorId, dims: Dims, offs: usize) -> TensorId {
        let src = &self[src_id];

        // Find the base tensor. If the src is a view of another tensor, that other
        // tensor's
        let (base_id, buf_id) = if let Some(parent_view) = src.view() {
            (parent_view.origin, self[parent_view.origin].buf_id)
        } else {
            (src_id, src.buf_id)
        };
        let data_offs = src.data_offs + offs;

        let op = Op::Unary {
            op: UnaryOp::Noop(Noop::View),
            a: src_id,
        };
        let layout = Layout::new(dims, src.typ());
        let base_size = self[base_id].size();
        op_assert!(
            offs + layout.size() <= base_size,
            "tensor '{src}' view_{n}d offset out of bounds (size={size} base_size={base_size})",
            src = src.name,
            n = layout.n(),
            size = offs + layout.size()
        );

        self.new_tensor_impl(
            layout,
            op,
            self.get_op_name(op),
            // Some(format!("{} (view)", src.name).into()),
            buf_id,
            data_offs,
            Some(View {
                origin: base_id,
                offs,
            }),
            None,
        )
    }

    /// Scale a 1D tensor by a scalar.
    ///
    /// The returned tensor is a copy of the input tensor.
    #[inline(always)]
    pub fn scale(&mut self, a_id: TensorId, scalar_id: TensorId) -> Result<TensorId, Oom> {
        self.scale_impl(a_id, scalar_id, Dup)
    }

    /// Scale a 1D tensor by a scalar.
    ///
    /// The returned tensor is a view of the input tensor and therefore its underlying buffer is
    /// modified.
    #[inline(always)]
    pub fn scale_inplace(&mut self, a_id: TensorId, scalar_id: TensorId) -> TensorId {
        self.scale_impl(a_id, scalar_id, Inplace)
            .expect(INPLACE_EXPECT)
    }

    fn scale_impl(
        &mut self,
        a_id: TensorId,
        scalar_id: TensorId,
        mode: Mode,
    ) -> Result<TensorId, Oom> {
        let a = &self[a_id];
        let scalar = &self[scalar_id];
        op_assert!(
            scalar.is_scalar(),
            "scalar tensor '{scalar}' must be a scalar (single element 1D tensor)",
            scalar = scalar.name,
        );
        op_assert!(
            a.is_padded_1d(),
            "tensor '{a}' must be padded 1D",
            a = a.name
        );
        op_assert!(
            a.is_contiguous(),
            "tensor '{a}' must contiguous",
            a = a.name
        );

        self.new_op_based_on_tensor(
            a_id,
            Op::Binary {
                op: BinaryOp::Scale,
                a: a_id,
                b: scalar_id,
            },
            mode,
        )
    }

    /// Sets elements above the diagonal to -INF.
    ///
    /// The returned tensor is a copy of the input tensor.
    #[inline(always)]
    pub fn diag_mask_inf(&mut self, a_id: TensorId, n_past: usize) -> Result<TensorId, Oom> {
        self.diag_mask_inf_impl(a_id, n_past, Dup)
    }

    /// Sets elements above the diagonal to -INF.
    /// TODO document n_past
    ///
    /// The returned tensor is a view of the input tensor and therefore its underlying buffer is
    /// modified.
    #[inline(always)]
    pub fn diag_mask_inf_inplace(&mut self, a_id: TensorId, n_past: usize) -> TensorId {
        self.diag_mask_inf_impl(a_id, n_past, Inplace)
            .expect(INPLACE_EXPECT)
    }

    fn diag_mask_inf_impl(
        &mut self,
        a_id: TensorId,
        n_past: usize,
        mode: Mode,
    ) -> Result<TensorId, Oom> {
        self.new_op_based_on_tensor(
            a_id,
            Op::Unary {
                op: UnaryOp::DiagMaskInf { n_past },
                a: a_id,
            },
            mode,
        )
    }

    ///
    /// The returned tensor is a copy of the input tensor.
    #[inline(always)]
    pub fn softmax(&mut self, a_id: TensorId) -> Result<TensorId, Oom> {
        self.softmax_impl(a_id, None, 1., Dup)
    }

    ///
    /// The returned tensor is a view of the input tensor and therefore its underlying buffer is
    /// modified.
    #[inline(always)]
    pub fn softmax_inplace(&mut self, a_id: TensorId) -> TensorId {
        self.softmax_impl(a_id, None, 1., Inplace)
            .expect(INPLACE_EXPECT)
    }

    fn softmax_impl(
        &mut self,
        a_id: TensorId,
        mask_id: Option<TensorId>,
        scale: f32,
        mode: Mode,
    ) -> Result<TensorId, Oom> {
        self.new_op_based_on_tensor(
            a_id,
            Op::Unary {
                op: UnaryOp::Softmax {
                    scale,
                    mask: mask_id,
                },
                a: a_id,
            },
            mode,
        )
    }

    ///
    /// The returned tensor is a copy of the input tensor.
    #[inline(always)]
    pub fn silu(&mut self, a_id: TensorId) -> Result<TensorId, Oom> {
        self.silu_impl(a_id, Dup)
    }

    ///
    /// The returned tensor is a view of the input tensor and therefore its underlying buffer is
    /// modified.
    #[inline(always)]
    pub fn silu_inplace(&mut self, a_id: TensorId) -> TensorId {
        self.silu_impl(a_id, Inplace).expect(INPLACE_EXPECT)
    }

    fn silu_impl(&mut self, a_id: TensorId, mode: Mode) -> Result<TensorId, Oom> {
        self.new_op_based_on_tensor(
            a_id,
            Op::Unary {
                op: UnaryOp::Silu,
                a: a_id,
            },
            mode,
        )
    }

    ///
    /// The returned tensor is a copy of the input tensor.
    #[inline(always)]
    pub fn relu(&mut self, a_id: TensorId) -> Result<TensorId, Oom> {
        self.relu_impl(a_id, Dup)
    }

    ///
    /// The returned tensor is a view of the input tensor and therefore its underlying buffer is
    /// modified.
    #[inline(always)]
    pub fn relu_inplace(&mut self, a_id: TensorId) -> TensorId {
        self.relu_impl(a_id, Inplace).expect(INPLACE_EXPECT)
    }

    fn relu_impl(&mut self, a_id: TensorId, mode: Mode) -> Result<TensorId, Oom> {
        self.new_op_based_on_tensor(
            a_id,
            Op::Unary {
                op: UnaryOp::Relu,
                a: a_id,
            },
            mode,
        )
    }

    /// This is the same as conv_1d with padding set to half.
    ///
    /// Alias for `self.conv_1d(input_id, kernel_id, stride, input.cols() / 2, dilation)`
    pub fn conv_1d_ph(
        &mut self,
        input_id: TensorId,
        kernel_id: TensorId,
        stride: usize,
        dilation: usize,
    ) -> Result<TensorId, Oom> {
        let input = &self[input_id];
        self.conv_1d(input_id, kernel_id, stride, input.cols() / 2, dilation)
    }

    // TODO: this and im2col are super complex, need to verify again
    pub fn conv_1d(
        &mut self,
        input_id: TensorId,
        kernel_id: TensorId,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<TensorId, Oom> {
        let input = &self[input_id];
        let kernel = &self[kernel_id];
        // TODO: still needed?
        op_assert!(kernel.is_matrix());
        op_assert_eq!(
            input.rows(),
            kernel.rows(),
            "input tensor '{input}' and kernel tensor '{kernel}' must have the same number of rows",
            input = input.name,
            kernel = kernel.name
        );
        // TODO: verify order here as it was taken from the ggml-metal.m
        op_assert_eq!(
            input.typ(),
            F16,
            "input tensor '{input}' must be F16",
            input = input.name
        );
        op_assert_eq!(
            kernel.typ(),
            F32,
            "kernel tensor '{kernel}' must be F32",
            kernel = kernel.name
        );

        let input_dims = input.dims;

        let im2col_id = self.im2col(
            input_id,
            kernel_id,
            Im2Col {
                stride0: stride,
                stride1: 0,
                padding0: padding,
                padding1: 0,
                dilation0: dilation,
                dilation1: 0,
            },
            false,
        )?;
        let im2col_dims = self[im2col_id].dims;

        // NB: The order of the dimension arguments is reversed in the reshape calls compared to the
        // ggml implementation, because ggml specifies columns first, while we specify rows first.
        let a = self.reshape(
            im2col_id,
            D2 {
                rows: im2col_dims[2] * im2col_dims[1],
                cols: im2col_dims[0],
            },
            // Dims::new_2d(im2col_dims[2] * im2col_dims[1], im2col_dims[0]),
        );
        let b = self.reshape(
            input_id,
            D2 {
                rows: input_dims[2],
                cols: input_dims[0] * input_dims[1],
            },
            // Dims::new_2d(input_dims[2], input_dims[0] * input_dims[1]),
        );

        let res = self.matmul(a, b)?;

        Ok(self.reshape(
            res,
            D3 {
                depth: im2col_dims[2],
                rows: im2col_dims[2],
                cols: im2col_dims[1],
            }, // Dims::new_3d(im2col_dims[2], im2col_dims[2], im2col_dims[1]),
        ))
    }

    pub fn im2col(
        &mut self,
        input_id: TensorId,
        kernel_id: TensorId,
        Im2Col {
            stride0,
            stride1,
            padding0,
            padding1,
            dilation0,
            dilation1,
        }: Im2Col,
        is_2d: bool,
    ) -> Result<TensorId, Oom> {
        let input = &self[input_id];
        let kernel = &self[kernel_id];
        if is_2d {
            op_assert_eq!(
                input.dims[2], kernel.dims[2],
                "input tensor '{input}' and kernel tensor '{kernel}' must have the same depth (dims[2])",
                input = input.name,
                kernel = kernel.name
            );
        } else {
            op_assert_eq!(
                input.rows(), kernel.rows(),
                "input tensor '{input}' and kernel tensor '{kernel}' must have the same number of rows",
                input = input.name,
                kernel = kernel.name
            );
        }

        fn get_conv_output_size(
            kernel_size: usize,
            input_size: usize,
            stride: usize,
            padding: usize,
            dilation: usize,
        ) -> usize {
            (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        }

        let ow = get_conv_output_size(kernel.cols(), input.cols(), stride0, padding0, dilation0);
        let layout = if is_2d {
            let oh =
                get_conv_output_size(kernel.rows(), input.rows(), stride1, padding1, dilation1);
            Layout::new_raw_dims(
                &[
                    input.dims[2] * input.dims[1] * input.dims[0],
                    ow,
                    oh,
                    kernel.dims[3],
                ],
                F32,
            )
        } else {
            Layout::new_raw_dims(&[input.dims[1] * input.dims[0], ow, kernel.dims[2], 1], F32)
        };

        // let buf = self.alloc.alloc(ToAllocArg::<A>::to_alloc_arg(&layout))?;
        let buf = self.alloc.alloc(A::Arg::from(&layout))?;
        let op = Op::Binary {
            op: BinaryOp::Im2Col(Im2Col {
                stride0,
                stride1,
                padding0,
                padding1,
                dilation0,
                dilation1,
            }),
            a: input_id,
            b: kernel_id,
        };
        Ok(self.new_tensor_impl(
            layout,
            op,
            self.get_op_name(op),
            buf.id,
            buf.offs,
            None,
            None,
        ))
    }

    /*
    struct ggml_tensor * ggml_get_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * idx) {
    GGML_ASSERT(a->ne[2] == idx->ne[1]);
    GGML_ASSERT(idx->ne[3] == 1);
    GGML_ASSERT(idx->type == GGML_TYPE_I32);

    bool is_node = false;

    if (a->grad || idx->grad) {
        is_node = true;
    }

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], idx->ne[0]);
    struct ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a->ne[0], idx->ne[0], idx->ne[1], idx->ne[2]);

    result->op   = GGML_OP_GET_ROWS;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = idx;
    */

    pub fn get_rows(&mut self, a_id: TensorId, idx_id: TensorId) -> Result<TensorId, Oom> {
        let a = &self[a_id];
        let idx = &self[idx_id];
        op_assert!(
            a.depth() == idx.rows(),
            "tensor '{a}' depth must match index tensor '{idx}' rows",
            a = a.name,
            idx = idx.name,
        );
        op_assert!(
            idx.batches() == 1,
            "index tensor '{idx}' must be at most 3D",
            idx = idx.name,
        );
        op_assert!(
            idx.typ() == I32,
            "index tensor '{idx}' must be F32",
            idx = idx.name,
        );

        let op = Op::Binary {
            op: BinaryOp::GetRows,
            a: a_id,
            b: idx_id,
        };
        let mut layout = Layout::new(
            D4 {
                batches: idx.depth(),
                depth: idx.rows(),
                rows: idx.cols(),
                cols: a.cols(),
            },
            F32,
        );
        // TODO: I think this is correct?
        layout.n = a.n;
        // let layout = Layout::new_2d(idx.cols(), a.cols(), F32);
        // let buf = self.alloc.alloc(ToAllocArg::<A>::to_alloc_arg(&layout))?;
        let buf = self.alloc.alloc(A::Arg::from(&layout))?;
        Ok(self.new_tensor_impl(
            layout,
            op,
            self.get_op_name(op),
            buf.id,
            buf.offs,
            None,
            None,
        ))
    }
}

// ---- helpers ----
impl<A: TensorAlloc, S: TensorSrc> TensorBuilder<A, S>
where
    // Layout: ToAllocArg<A>,
    A::Arg: for<'a> From<&'a Layout>,
{
    fn new_op_based_on_tensor(
        &mut self,
        id: TensorId,
        op: Op,
        mode: Mode,
    ) -> Result<TensorId, Oom> {
        let tensor = &self[id];
        let layout = tensor.layout;
        let (buf_id, data_offs, view) = if mode == Inplace {
            // Find the base tensor. If the src is a view of another tensor, that other
            // tensor's
            let (base_id, buf_id) = if let Some(parent_view) = tensor.view() {
                (parent_view.origin, self[parent_view.origin].buf_id)
            } else {
                (id, tensor.buf_id)
            };
            (
                buf_id,
                tensor.data_offs,
                Some(View {
                    origin: base_id,
                    offs: 0,
                }),
            )
        } else {
            let buf = self.alloc.alloc(A::Arg::from(&layout))?;
            (buf.id, buf.offs, None)
        };
        Ok(self.new_tensor_impl(
            layout,
            op,
            self.get_op_name(op),
            buf_id,
            data_offs,
            view,
            None,
        ))
    }

    fn get_op_name(&self, op: Op) -> Option<Cow<'static, str>> {
        match self.auto_op_names {
            AutoOpName::Disabled => None,
            AutoOpName::Simple => Some(Cow::Owned(op.to_string())),
            AutoOpName::Expanded { with_id } => {
                fn get_name(t: &Tensor, id: TensorId, with_id: bool) -> Cow<'_, str> {
                    if with_id || t.name.is_empty() {
                        Cow::Owned(id.to_string())
                    } else {
                        Cow::Borrowed(t.name.deref())
                    }
                }
                // Get the source tensor names for the expanded op name.
                Some(match op {
                    Op::Const => "const".into(),
                    Op::Unary { op, a: x } => {
                        Cow::Owned(format!("{op}({x})", x = get_name(&self[x], x, with_id)))
                    }
                    Op::Binary { op, a: x, b: y } => Cow::Owned(format!(
                        "({x} {op} {y})",
                        x = get_name(&self[x], x, with_id),
                        y = get_name(&self[y], y, with_id),
                    )),
                })
            }
        }
    }
}

const INPLACE_EXPECT: &str = "inplace operation cannot allocate";

// Checks if `expected` can be represented as a repeatition of `to_repeat`.
fn can_repeat(to_repeat: &Tensor, expected: &Tensor) -> bool {
    (expected.dims[0] % to_repeat.dims[0] == 0)
        && (expected.dims[1] % to_repeat.dims[1] == 0)
        && (expected.dims[2] % to_repeat.dims[2] == 0)
        && (expected.dims[3] % to_repeat.dims[3] == 0)
}

#[cfg(test)]
mod tests {
    #![allow(unused)]

    use super::*;

    #[test]
    fn calcs_correct_data_offs_from_view_src() {
        let mut tb = TensorBuilder::prealloc(1024).unwrap();

        // Alloc tensor so that the data_offs is not 0.
        let pad = tb.new_tensor(D1 { cols: 10 }).unwrap();

        let offs = |n| n * F32.size();

        let grandparent = tb.new_tensor(D1 { cols: 10 }).unwrap();
        let parent = tb.view_1d(grandparent, D1 { cols: 7 }, 3);
        let child = tb.view_1d(parent, D1 { cols: 5 }, 2);

        assert_eq!(tb[grandparent].data_offs, offs(10));
        assert!(tb[grandparent].view.is_none());

        assert_eq!(tb[parent].data_offs, offs(10) + 3);
        assert_eq!(
            tb[parent].view,
            Some(View {
                origin: grandparent,
                offs: 3
            })
        );

        assert_eq!(tb[child].data_offs, offs(10) + 3 + 2);
        // The child's view origin is still the grandparent, the original tensor.
        assert_eq!(
            tb[child].view,
            Some(View {
                origin: grandparent,
                offs: 2
            })
        );
    }
}
