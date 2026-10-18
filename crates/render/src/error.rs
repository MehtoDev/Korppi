use std::fmt;

/// Errors emitted by the rendering engine and Render Graph compiler.
#[derive(Debug)]
pub enum RenderError {
    /// A cyclic dependency was detected in the render graph.
    CyclicDependency {
        /// Name of the pass involved in the cycle.
        pass_name: String,
    },

    /// Resource was accessed with an invalid or expired handle.
    InvalidHandle(String),

    /// Missing dependency or root target.
    ValidationError(String),

    /// Underlying RHI hardware error.
    Rhi(korppi_rhi::RhiError),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CyclicDependency { pass_name } => {
                write!(
                    f,
                    "Cyclic dependency detected in render graph pass: {pass_name}"
                )
            }
            Self::InvalidHandle(msg) => {
                write!(f, "Invalid resource handle in render graph: {msg}")
            }
            Self::ValidationError(msg) => write!(f, "Render graph validation failed: {msg}"),
            Self::Rhi(err) => write!(f, "RHI hardware error: {err}"),
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rhi(err) => Some(err),
            _ => None,
        }
    }
}

impl From<korppi_rhi::RhiError> for RenderError {
    fn from(err: korppi_rhi::RhiError) -> Self {
        Self::Rhi(err)
    }
}
