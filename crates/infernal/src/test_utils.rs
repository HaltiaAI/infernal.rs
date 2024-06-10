use tracing::trace;

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
