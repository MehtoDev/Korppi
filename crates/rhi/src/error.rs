use std::fmt;

/// Core error type for Render Hardware Interface operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RhiError {
    /// RHI device initialization failed.
    InitializationFailed(String),

    /// GPU device was lost or removed.
    DeviceLost,

    /// Out of device or host memory.
    OutOfMemory(String),

    /// Requested resource handle is invalid or was destroyed.
    InvalidHandle(String),

    /// Validation error when creating a pipeline or recording commands.
    ValidationError(String),

    /// Requested hardware feature or format is unsupported on this device.
    Unsupported(String),

    /// Shader compilation or bytecode parsing failed.
    ShaderCompilationFailed(String),

    /// Mapping or accessing GPU buffer memory failed.
    BufferMappingFailed(String),

    /// Swapchain acquisition, resizing, or presentation failed.
    SwapchainError(String),

    /// Backend-specific driver or API error.
    BackendError(String),
}

impl fmt::Display for RhiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitializationFailed(msg) => write!(f, "RHI device initialization failed: {msg}"),
            Self::DeviceLost => write!(f, "RHI graphics device lost"),
            Self::OutOfMemory(msg) => write!(f, "RHI out of memory: {msg}"),
            Self::InvalidHandle(msg) => write!(f, "RHI invalid resource handle: {msg}"),
            Self::ValidationError(msg) => write!(f, "RHI validation error: {msg}"),
            Self::Unsupported(msg) => write!(f, "RHI feature unsupported: {msg}"),
            Self::ShaderCompilationFailed(msg) => {
                write!(f, "RHI shader compilation failed: {msg}")
            }
            Self::BufferMappingFailed(msg) => write!(f, "RHI buffer mapping failed: {msg}"),
            Self::SwapchainError(msg) => write!(f, "RHI swapchain error: {msg}"),
            Self::BackendError(msg) => write!(f, "RHI backend error: {msg}"),
        }
    }
}

impl std::error::Error for RhiError {}
