//! Diagnostic logging and tracing subscriber initialization.

/// Initializes the global tracing subscriber with the default filter level.
///
/// # Examples
/// ```
/// korppi_core::init_logging("info");
/// ```
#[cfg(feature = "logging")]
pub fn init_logging(default_level: &str) {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .try_init();
}

/// No-op fallback when the `logging` feature is not enabled.
#[cfg(not(feature = "logging"))]
pub fn init_logging(_default_level: &str) {}
