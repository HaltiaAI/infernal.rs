#![allow(unused)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use infernal::{
    builder::{AutoOpName, TensorOptionsExt as _},
    Backend as _, ComputeGraph, Layout, TensorBuilder, TensorCtx, Type,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let Ctx {
        mut backend,
        // mut gf,
        tensor_size,
        exp_mem_size,
        // } = setup().unwrap();
    } = todo!();

    let mut g = c.benchmark_group("large-matmul");
    g.sample_size(20); // default 100 runs for too long

    // All tensors can fit in RAM.
    g.bench_function("no-mem-limit", |b| {
        b.iter(|| {
            // backend.compute_graph(black_box(&gf)).unwrap();
        })
    });

    // Most tensors can fit in RAM.
    backend.set_mem_limit(exp_mem_size - 1024).unwrap();
    // gf.find_exec_plan(backend.mem_limit()).unwrap();
    g.bench_function("modest-mem-limit", |b| {
        b.iter(|| {
            // backend.compute_graph(black_box(&gf)).unwrap();
        })
    });

    // Only the current tensor and its sources can fit in RAM.
    backend.set_mem_limit(3 * tensor_size).unwrap();
    // gf.find_exec_plan(backend.mem_limit()).unwrap();
    g.bench_function("aggressive-mem-limit", |b| {
        b.iter(|| {
            // backend.compute_graph(black_box(&gf)).unwrap();
        })
    });

    g.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

struct Ctx {
    backend: infernal::backend::Metal,
    // gf: ComputeGraph<TensorCtx>,
    tensor_size: usize,
    exp_mem_size: usize,
}

// fn setup() -> anyhow::Result<Ctx> {
//     let mut backend = infernal::backend::Metal::new()?;

//     let n_tensors = 6; // has to be >4
//     let one_side_size = 8192; // ~256 MiB per tensor
//     let layout = Layout::new_2d(one_side_size, one_side_size, Type::F32);
//     let exp_mem_size = n_tensors * layout.size();
//     println!("Single tensor size: {} MiB", layout.size() / 1024 / 1024);
//     println!(
//         "Expected total RAM usage: {} MiB",
//         n_tensors * layout.size() / 1024 / 1024
//     );

//     let mut tb = TensorBuilder::alloc()
//         .with_name("large_matmul_test")
//         .with_auto_op_names(AutoOpName::Expanded { with_id: false });

//     let x = tb.new_tensor(layout.with_name("x"))?;
//     let y = tb.new_tensor(layout.with_name("y"))?;
//     tb.fill_2d_f32(x, 1.3);
//     tb.fill_2d_f32(y, 9.5);

//     let mut i = tb.add(x, y)?;
//     for _ in 0..n_tensors - 4 {
//         i = tb.add(y, i)?;
//     }
//     let res = tb.matmul(y, i)?;

//     let (ctx, _) = tb.finish(&mut backend)?;

//     let mut gf = ComputeGraph::from(ctx);
//     // gf.expand_forward(res)?;
//     // gf.find_exec_plan(backend.mem_limit())?;

//     Ok(Ctx {
//         backend,
//         gf,
//         tensor_size: layout.size(),
//         exp_mem_size,
//     })
// }
