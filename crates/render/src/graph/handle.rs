/// Opaque virtual handle to a texture managed by the Render Graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderGraphTextureHandle(pub(crate) u32);

impl RenderGraphTextureHandle {
    /// Returns the raw virtual index of this handle.
    #[inline]
    pub fn id(&self) -> u32 {
        self.0
    }
}

/// Opaque virtual handle to a buffer managed by the Render Graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderGraphBufferHandle(pub(crate) u32);

impl RenderGraphBufferHandle {
    /// Returns the raw virtual index of this handle.
    #[inline]
    pub fn id(&self) -> u32 {
        self.0
    }
}
