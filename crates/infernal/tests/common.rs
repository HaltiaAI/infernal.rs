#![allow(dead_code)]

use infernal::{buf::SliceCast, Backend, Tensor, TensorId, TensorSrc};

pub const KB: usize = 1024;
pub const MB: usize = 1024 * KB;

// /// Helper macro for quickly instantiating 2D tensors with a given type and value.
// #[macro_export]
// macro_rules! t {
//     ($tb:expr => $({ $tensor:tt = [$rows:expr, $cols:expr] $(@ $type:ident)? $(= $val:expr)? }),* $(,)?) => {
//         $(
//             t!(recursion_end $tb => { $tensor = [$rows, $cols] $(@ $type)? $(= $val)? });
//         )*
//     };
//     (recursion_end $tb:expr => $({ $tensor:tt = [$rows:expr, $cols:expr] }),* $(,)?) => {
//         $(
//             let $tensor = $tb.new_tensor(
//                 Layout::new_2d($rows, $cols, infernal::Type::F32).with_name(stringify!($tensor)),
//             )?;
//         )*
//     };
//     (recursion_end $tb:expr => $({ $tensor:tt = [$rows:expr, $cols:expr] @ $type:ident }),* $(,)?) => {
//         $(
//             let $tensor = $tb.new_tensor(
//                 Layout::new_2d($rows, $cols, infernal::Type::$type).with_name(stringify!($tensor)),
//             )?;
//         )*
//     };
//     (recursion_end $tb:expr => $({ $tensor:tt = [$rows:expr, $cols:expr] = $val:expr }),* $(,)?) => {
//         $(
//             let $tensor = $tb.new_tensor(
//                 Layout::new_2d($rows, $cols, infernal::Type::F32).with_name(stringify!($tensor)),
//             )?;
//             $tb.fill_2d_tensor($tensor, $val);
//         )*
//     };
//     (recursion_end $tb:expr => $({ $tensor:tt = [$rows:expr, $cols:expr] @ $type:ident = $val:expr }),* $(,)?) => {
//         $(
//             let $tensor = $tb.new_tensor(
//                 Layout::new_2d($rows, $cols, infernal::Type::$type).with_name(stringify!($tensor)),
//             )?;
//             $tb.fill_2d_tensor($tensor, $val);
//         )*
//     };
// }

/// Initializes trace logging once.
pub fn init_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            // .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .init();
        //
        // let subscriber = tracing_subscriber::registry()
        //     .with(EnvFilter::from_default_env())
        //     .with(tracing_subscriber::fmt::layer().with_writer(std::io::stdout));
        // let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

pub fn assert_elements(
    ctx: &impl TensorSrc,
    result: TensorId,
    expected: TensorId,
    backend: &impl Backend,
) {
    let result = ctx.get_tensor(result).expect("'result' tensor not found");
    let expected = ctx
        .get_tensor(expected)
        .expect("'expected' tensor not found");
    assert_eq!(result.layout(), expected.layout());
    print_rows(result, backend);
    assert_eq!(
        result.elements::<f32>(backend),
        expected.elements::<f32>(backend)
    );
}

pub fn print_rows(t: &Tensor, backend: &impl Backend) {
    let row_stride = t.row_stride();
    let rows = t.rows();
    let cols = t.cols();
    println!("rows: {}, cols: {}", rows, cols);
    let buf = backend.get_tensor_buf(t).unwrap().borrow();
    // println!("buf = {:?}", &buf[..]);
    let data = buf.tensor_data(t);
    for row in 0..rows {
        let row = &data[row * row_stride..row * row_stride + cols];
        let row: &[f32] = row.cast();
        println!("row = {:?}", row);
    }
}

pub fn print_tensor_meta(t: &Tensor) {
    println!("============================");
    println!("tensor '{}' metadata:", t.name());
    println!("layout: {:?}", t.layout());
    println!("layout size: {:?}", t.size());
    println!("layout total_elems: {:?}", t.total_elems());
    println!("layout is_transposed: {:?}", t.is_transposed());
    println!("layout is_padded_1d: {:?}", t.is_padded_1d());
    println!("============================");
}
