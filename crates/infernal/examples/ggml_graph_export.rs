use std::{
    borrow::Cow,
    ffi::{CStr, CString},
    fs::File,
    io::Write,
    os::unix::ffi::OsStrExt,
};

use infernal::{
    graph::inspect::{
        export_debug, Graph, Node, NodeInfo, OpSimple, ParentIdx, ParentNode, WeightInfo,
    },
    op::{BinaryOpDiscriminants, NoopDiscriminants, UnaryOpDiscriminants},
    Layout, Type,
};
use libc::c_void;
use llama_cpp_sys::ggml;

const INPUT: &str = "why is the sky blue?";

fn main() -> anyhow::Result<()> {
    let path = dirs::home_dir()
        .expect("$HOME must be set")
        .join(".config/haltia/default.gguf");

    let graph = GgmlGraph::new(path)?;

    // TODO: specify the output file path as a CLI arg
    let mut file = File::create("ggml-graph.json")?;
    let mut writer = std::io::BufWriter::new(&mut file);
    export_debug(&graph, &mut writer)?;

    writer.flush()?;

    println!("Successfully exported ggml debug graph.");

    Ok(())
}

pub struct GgmlGraph {
    model: *mut llama_cpp_sys::llama_model,
    ctx: *mut llama_cpp_sys::llama_context,
    graph: llama_cpp_sys::ggml_cgraph,
    mmap_ptr: *mut c_void,
}

impl GgmlGraph {
    fn new(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path_cstr = CString::new(path.as_ref().as_os_str().as_bytes())?;

        unsafe {
            let params = llama_cpp_sys::llama_context_default_params();
            let model = llama_cpp_sys::llama_load_model_from_file(path_cstr.as_ptr(), params);
            let ctx = llama_cpp_sys::llama_new_context_with_model(model, params);

            let mut tokens = vec![0; 100];
            let input = CString::new(INPUT).unwrap();
            let n_tokens = llama_cpp_sys::llama_tokenize_with_model(
                model,
                input.as_ptr(),
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
            );
            let tokens = &tokens[..n_tokens as usize];

            // Need to run eval once to materialize all tensors.
            let fail = llama_cpp_sys::llama_eval(ctx, tokens.as_ptr(), tokens.len() as _, 0, 8);
            assert_eq!(fail, 0);

            let graph = *llama_cpp_sys::infernal_ext_get_last_cgraph();
            println!(
                "n_nodes={} n_leafs={} tokens={tokens:?}",
                graph.n_nodes, graph.n_leafs
            );

            #[derive(serde::Serialize)]
            pub struct GgmlCgraphExport {
                pub nodes: Vec<String>,
                pub leafs: Vec<String>,
            }

            let mut cgraph_export = GgmlCgraphExport {
                nodes: Vec::new(),
                leafs: Vec::new(),
            };

            for node in graph.nodes.iter().take(graph.n_nodes as _) {
                let name = cstr_to_string(&node.as_ref().unwrap().name);
                cgraph_export.nodes.push(name);
            }

            std::fs::write(
                "ggml-cgraph-nodes.json",
                serde_json::to_vec(&cgraph_export)?,
            )?;

            let mmap_ptr = llama_cpp_sys::infernal_ext_llama_mapping(ctx);

            Ok(Self {
                model,
                ctx,
                graph,
                mmap_ptr,
            })
        }
    }
}

impl Drop for GgmlGraph {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys::llama_free(self.ctx);
            llama_cpp_sys::llama_free_model(self.model);
        }
    }
}

impl Graph for GgmlGraph {
    fn n_nodes(&self) -> usize {
        self.graph.n_nodes as usize
    }

    fn n_leaf_nodes(&self) -> usize {
        self.graph.n_leafs as usize
    }

    fn get_node(&self, idx: usize) -> Option<Node> {
        if idx >= self.n_nodes() {
            return None;
        }

        let node = unsafe { self.graph.nodes[idx].as_ref().unwrap() };

        Some(Node {
            info: NodeInfo {
                id: None, // infernal specific
                name: cstr_to_string(&node.name[..]),
                op: convert_op(node),
                layout: convert_layout(node),
            },
            parents: self.convert_parents(node),
        })
    }
}

impl GgmlGraph {
    fn convert_parents(&self, t: &llama_cpp_sys::ggml_tensor) -> Vec<ParentNode> {
        let mut parents = Vec::new();
        for src in &t.src {
            if !src.is_null() {
                let idx = if let Some(idx) = self.graph.nodes.iter().position(|t| *t == *src) {
                    ParentIdx::Node(idx)
                } else {
                    let idx = self
                        .graph
                        .leafs
                        .iter()
                        .position(|t| *t == *src)
                        .expect("tensor src not in graph");
                    ParentIdx::Leaf(idx)
                };

                let src = unsafe { &**src };
                let name = cstr_to_string(&src.name[..]);
                let is_weight = name.ends_with(".weight");

                parents.push(ParentNode {
                    idx,
                    info: NodeInfo {
                        id: None,
                        name,
                        op: convert_op(src),
                        layout: convert_layout(src),
                    },
                    data: WeightInfo::new(
                        convert_data(src),
                        if is_weight {
                            Some(unsafe { src.data.offset_from(self.mmap_ptr) } as usize)
                        } else {
                            None
                        },
                    ),
                });
            }
        }
        parents
    }
}

fn cstr_to_string(name: &[i8]) -> String {
    unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .to_string()
}

fn convert_layout(t: &llama_cpp_sys::ggml_tensor) -> Layout {
    #[allow(clippy::needless_range_loop)]
    // Convert dims to an usize array.
    let mut dims = [0; 4];
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        dims[i] = t.ne[i] as usize;
    }
    let typ = convert_type(t.type_);
    let layout = Layout::from_raw(t.n_dims as usize, dims, t.nb, typ);
    unsafe {
        assert_eq!(
            layout.size(),
            ggml::ggml_nbytes(std::mem::transmute(t as *const _)) as usize,
            "Layout::size does not match size reported by ggml, this is a bug."
        );
    }
    layout
}

fn convert_data(t: &llama_cpp_sys::ggml_tensor) -> &[u8] {
    unsafe {
        assert!(!t.data.is_null());
        std::slice::from_raw_parts(
            t.data as *const u8,
            ggml::ggml_nbytes(std::mem::transmute(t as *const _)),
        )
    }
}

fn convert_type(typ: ggml::ggml_type) -> Type {
    match typ {
        ggml::ggml_type_GGML_TYPE_F32 => Type::F32,
        ggml::ggml_type_GGML_TYPE_F16 => Type::F16,
        // ggml::ggml_type_GGML_TYPE_Q4_0 = 2,
        // ggml::ggml_type_GGML_TYPE_Q4_1 = 3,
        // // GGML_TYPE_Q4_2 = 4, support has been removed
        // // GGML_TYPE_Q4_3 (5) support has been removed
        // ggml::ggml_type_GGML_TYPE_Q5_0 = 6,
        // ggml::ggml_type_GGML_TYPE_Q5_1 = 7,
        // ggml::ggml_type_GGML_TYPE_Q8_0 = 8,
        // ggml::ggml_type_GGML_TYPE_Q8_1 = 9,
        // k-quantizations
        ggml::ggml_type_GGML_TYPE_Q2_K => Type::Q2_K,
        ggml::ggml_type_GGML_TYPE_Q3_K => Type::Q3_K,
        ggml::ggml_type_GGML_TYPE_Q4_K => Type::Q4_K,
        ggml::ggml_type_GGML_TYPE_Q5_K => Type::Q5_K,
        ggml::ggml_type_GGML_TYPE_Q6_K => Type::Q6_K,
        // ggml::ggml_type_GGML_TYPE_Q8_K => Type::Q8_K,
        // ggml::ggml_type_GGML_TYPE_I8 => Type::I8,
        // ggml::ggml_type_GGML_TYPE_I16 => Type::I16,
        ggml::ggml_type_GGML_TYPE_I32 => Type::I32,
        _ => unimplemented!("ggml type {:?} is not supported by infernal", unsafe {
            CStr::from_ptr(ggml::ggml_type_name(typ))
        }),
    }
}

fn convert_op(t: &llama_cpp_sys::ggml_tensor) -> OpSimple {
    match t.op {
        ggml::ggml_op_GGML_OP_NONE => OpSimple::Const,

        ggml::ggml_op_GGML_OP_DUP | ggml::ggml_op_GGML_OP_CPY | ggml::ggml_op_GGML_OP_CONT => {
            OpSimple::Binary(BinaryOpDiscriminants::Copy)
        }

        ggml::ggml_op_GGML_OP_ADD => OpSimple::Binary(BinaryOpDiscriminants::Add),
        // ggml::ggml_op_GGML_OP_ADD1,
        // ggml::ggml_op_GGML_OP_ACC,
        // ggml::ggml_op_GGML_OP_SUB => OpSimple::Binary(BinaryOpDiscriminants::Sub),
        ggml::ggml_op_GGML_OP_MUL => OpSimple::Binary(BinaryOpDiscriminants::Mul),
        ggml::ggml_op_GGML_OP_DIV => OpSimple::Binary(BinaryOpDiscriminants::Div),
        // ggml::ggml_op_GGML_OP_SQR,
        // ggml::ggml_op_GGML_OP_SQRT,
        // ggml::ggml_op_GGML_OP_LOG,
        // ggml::ggml_op_GGML_OP_SUM,
        // ggml::ggml_op_GGML_OP_SUM_ROWS,
        // ggml::ggml_op_GGML_OP_MEAN,
        // ggml::ggml_op_GGML_OP_ARGMAX,
        // ggml::ggml_op_GGML_OP_REPEAT,
        // ggml::ggml_op_GGML_OP_REPEAT_BACK,
        // ggml::ggml_op_GGML_OP_CONCAT,
        // ggml::ggml_op_GGML_OP_SILU_BACK,
        // ggml::ggml_op_GGML_OP_NORM, // normalize
        ggml::ggml_op_GGML_OP_RMS_NORM => OpSimple::Unary(UnaryOpDiscriminants::RmsNorm),
        // ggml::ggml_op_GGML_OP_RMS_NORM_BACK,
        // ggml::ggml_op_GGML_OP_GROUP_NORM,
        ggml::ggml_op_GGML_OP_MUL_MAT => OpSimple::Binary(BinaryOpDiscriminants::Matmul),
        // ggml::ggml_op_GGML_OP_OUT_PROD,
        ggml::ggml_op_GGML_OP_SCALE => OpSimple::Binary(BinaryOpDiscriminants::Scale),
        // ggml::ggml_op_GGML_OP_SET,
        ggml::ggml_op_GGML_OP_RESHAPE => OpSimple::Noop(NoopDiscriminants::Reshape),
        ggml::ggml_op_GGML_OP_VIEW => OpSimple::Noop(NoopDiscriminants::View),
        ggml::ggml_op_GGML_OP_PERMUTE => OpSimple::Noop(NoopDiscriminants::Permute),

        ggml::ggml_op_GGML_OP_TRANSPOSE => OpSimple::Noop(NoopDiscriminants::Transpose),
        ggml::ggml_op_GGML_OP_GET_ROWS => OpSimple::Binary(BinaryOpDiscriminants::GetRows),
        // ggml::ggml_op_GGML_OP_GET_ROWS_BACK,
        // ggml::ggml_op_GGML_OP_DIAG,
        ggml::ggml_op_GGML_OP_DIAG_MASK_INF => OpSimple::Unary(UnaryOpDiscriminants::DiagMaskInf),
        // ggml::ggml_op_GGML_OP_DIAG_MASK_ZERO,
        ggml::ggml_op_GGML_OP_SOFT_MAX => OpSimple::Unary(UnaryOpDiscriminants::Softmax),
        // ggml::ggml_op_GGML_OP_SOFT_MAX_BACK,
        ggml::ggml_op_GGML_OP_ROPE => OpSimple::Unary(UnaryOpDiscriminants::Rope),
        // ggml::ggml_op_GGML_OP_ROPE_BACK,
        // ggml::ggml_op_GGML_OP_ALIBI,
        // ggml::ggml_op_GGML_OP_CLAMP,
        // ggml::ggml_op_GGML_OP_CONV_1D,
        // ggml::ggml_op_GGML_OP_CONV_2D,
        // ggml::ggml_op_GGML_OP_CONV_TRANSPOSE_2D,
        // ggml::ggml_op_GGML_OP_POOL_1D,
        // ggml::ggml_op_GGML_OP_POOL_2D,

        // ggml::ggml_op_GGML_OP_UPSCALE, // nearest interpolate

        // ggml::ggml_op_GGML_OP_FLASH_ATTN,
        // ggml::ggml_op_GGML_OP_FLASH_FF,
        // ggml::ggml_op_GGML_OP_FLASH_ATTN_BACK,
        // ggml::ggml_op_GGML_OP_WIN_PART,
        // ggml::ggml_op_GGML_OP_WIN_UNPART,
        // ggml::ggml_op_GGML_OP_GET_REL_POS,
        // ggml::ggml_op_GGML_OP_ADD_REL_POS,
        ggml::ggml_op_GGML_OP_UNARY => {
            unsafe {
                match ggml::ggml_get_unary_op(std::mem::transmute(t as *const _)) {
                    // ggml_op_GGML_UNARY_OP_ABS,
                    // ggml_op_GGML_UNARY_OP_SGN,
                    // ggml_op_GGML_UNARY_OP_NEG,
                    // ggml_op_GGML_UNARY_OP_STEP,
                    // ggml_op_GGML_UNARY_OP_TANH,
                    // ggml_op_GGML_UNARY_OP_ELU,
                    ggml::ggml_unary_op_GGML_UNARY_OP_RELU => {
                        OpSimple::Unary(UnaryOpDiscriminants::Relu)
                    }
                    ggml::ggml_unary_op_GGML_UNARY_OP_GELU => {
                        todo!()
                        // OpSimple::UnaryOp(UnaryOpDiscriminants::Gelu)
                    }
                    // ggml_op_GGML_UNARY_OP_GELU_QUICK,
                    ggml::ggml_unary_op_GGML_UNARY_OP_SILU => {
                        OpSimple::Unary(UnaryOpDiscriminants::Silu)
                    }
                    op => unimplemented!(
                        "ggml unary op {:?} is not supported by infernal",
                        CStr::from_ptr(ggml::ggml_op_name(op)),
                    ),
                }
            }
        }

        // ggml_op_GGML_OP_MAP_UNARY,
        // ggml_op_GGML_OP_MAP_BINARY,

        // ggml_op_GGML_OP_MAP_CUSTOM1_F32,
        // ggml_op_GGML_OP_MAP_CUSTOM2_F32,
        // ggml_op_GGML_OP_MAP_CUSTOM3_F32,

        // ggml_op_GGML_OP_MAP_CUSTOM1,
        // ggml_op_GGML_OP_MAP_CUSTOM2,
        // ggml_op_GGML_OP_MAP_CUSTOM3,

        // ggml_op_GGML_OP_CROSS_ENTROPY_LOSS,
        // ggml_op_GGML_OP_CROSS_ENTROPY_LOSS_BACK,

        // ggml_op_GGML_OP_COUNT,
        op => unimplemented!("ggml op {:?} is not supported by infernal", unsafe {
            CStr::from_ptr(ggml::ggml_op_name(op))
        }),
    }
}
