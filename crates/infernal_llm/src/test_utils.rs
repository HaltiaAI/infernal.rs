use std::path::PathBuf;

use infernal::backend::{self, Metal};
use tracing::trace;

use crate::{
    tokenizer::{spm::Spm, Tokenizer},
    DefaultModelRunner, Llama2, Model,
};

/// Initializes trace logging once.
pub(crate) fn init_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .init();
    });
    trace!("tracing initialized");
}

pub(crate) fn get_llama_runner() -> (Metal, DefaultModelRunner<Llama2>, PathBuf) {
    let (backend, runner, path) = get_model_runner::<Llama2>();
    println!("gpu layers: {:?}", runner.model().n_gpu_layers);
    (backend, runner, path)
}

pub(crate) fn get_model_runner<M: Model>() -> (Metal, DefaultModelRunner<M>, PathBuf) {
    let path = dirs::home_dir()
        .expect("$HOME must be set")
        .join(".config/haltia/default.gguf");
    let params = crate::Params::default();
    println!("params: {:#?}", params);

    let (model, tokenizer, tl) = crate::loader::load::<M>(&path, &params).unwrap();
    println!("arch: {:?}", Llama2::ARCH);
    println!("ftype: {:?}", model.common().ftype);
    println!("name: {:?}", model.common().name);
    println!("hparams: {:?}", model.common().hparams);
    println!("vocab special: {:?}", tokenizer.special());
    println!("n_layers: {:?}", model.common().hparams.n_layers);

    let mut backend = backend::Metal::new().unwrap();
    let runner = DefaultModelRunner::new(model, params, tokenizer, tl, &mut backend).unwrap();

    (backend, runner, path)
}
