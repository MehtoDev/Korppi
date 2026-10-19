/// Opaque virtual handle to a texture managed by the Render Graph (`RND-21`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderGraphTextureHandle {
    pub(crate) id: u32,
    pub(crate) graph_id: u32,
}

impl RenderGraphTextureHandle {
    #[inline]
    pub(crate) const fn new(id: u32, graph_id: u32) -> Self {
        Self { id, graph_id }
    }

    /// Constructs a raw handle for validation and testing (`RND-21`).
    #[inline]
    pub const fn from_raw(id: u32) -> Self {
        Self { id, graph_id: 0 }
    }

    /// Returns the raw virtual index of this handle.
    #[inline]
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// Returns the unique ID of the [`RenderGraph`](crate::graph::RenderGraph) that created this handle.
    #[inline]
    pub const fn graph_id(&self) -> u32 {
        self.graph_id
    }
}

/// Opaque virtual handle to a buffer managed by the Render Graph (`RND-21`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderGraphBufferHandle {
    pub(crate) id: u32,
    pub(crate) graph_id: u32,
}

impl RenderGraphBufferHandle {
    #[inline]
    pub(crate) const fn new(id: u32, graph_id: u32) -> Self {
        Self { id, graph_id }
    }

    /// Constructs a raw handle for validation and testing (`RND-21`).
    #[inline]
    pub const fn from_raw(id: u32) -> Self {
        Self { id, graph_id: 0 }
    }

    /// Returns the raw virtual index of this handle.
    #[inline]
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// Returns the unique ID of the [`RenderGraph`](crate::graph::RenderGraph) that created this handle.
    #[inline]
    pub const fn graph_id(&self) -> u32 {
        self.graph_id
    }
}
