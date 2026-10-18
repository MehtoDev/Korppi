use korppi_rhi::{
    AccessFlags, BufferHandle, BufferUsage, FrameIndex, ImageLayout, PipelineStageFlags,
    SampleCount, TextureDimension, TextureFormat, TextureHandle, TextureUsage, FRAMES_IN_FLIGHT,
};

/// Descriptor for creating a transient texture within the Render Graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TransientTextureDesc {
    /// Human-readable debug name.
    pub name: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Hardware texture format.
    pub format: TextureFormat,
    /// Allowed texture usages.
    pub usage: TextureUsage,
    /// MSAA sample count.
    pub sample_count: SampleCount,
    /// Dimensionality.
    pub dimension: TextureDimension,
}

impl TransientTextureDesc {
    /// Creates a 2D transient texture descriptor with default 1x sample count.
    ///
    /// Does not allocate on the heap for steady-state per-frame graph construction (RND-11).
    pub fn new(
        _name: impl AsRef<str>,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Self {
        Self {
            name: String::new(),
            width,
            height,
            format,
            usage,
            sample_count: SampleCount::Count1,
            dimension: TextureDimension::D2,
        }
    }

    /// Sets the MSAA sample count.
    pub fn with_samples(mut self, samples: SampleCount) -> Self {
        self.sample_count = samples;
        self
    }
}

/// Descriptor for creating a transient buffer within the Render Graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TransientBufferDesc {
    /// Human-readable debug name.
    pub name: String,
    /// Size in bytes.
    pub size: u64,
    /// Allowed buffer usages.
    pub usage: BufferUsage,
}

impl TransientBufferDesc {
    /// Creates a transient buffer descriptor without heap allocation (RND-11).
    pub fn new(_name: impl AsRef<str>, size: u64, usage: BufferUsage) -> Self {
        Self {
            name: String::new(),
            size,
            usage,
        }
    }
}

/// Access state requested by a pass for a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureAccess {
    /// Pipeline stages where the texture is accessed.
    pub stage: PipelineStageFlags,
    /// Access flags (reads, writes).
    pub access: AccessFlags,
    /// Desired Vulkan/OpenGL image layout.
    pub layout: ImageLayout,
}

/// Access state requested by a pass for a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferAccess {
    /// Pipeline stages where the buffer is accessed.
    pub stage: PipelineStageFlags,
    /// Access flags (reads, writes).
    pub access: AccessFlags,
}

/// Internal state tracking for a virtual texture in the Render Graph.
#[derive(Clone, Debug)]
pub(crate) enum VirtualTextureKind {
    Transient {
        desc: TransientTextureDesc,
        mip_levels: u32,
        array_layers: u32,
    },
    Imported {
        handle: TextureHandle,
        initial_access: TextureAccess,
        final_access: Option<TextureAccess>,
        width: u32,
        height: u32,
        mip_levels: u32,
        array_layers: u32,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct VirtualTexture {
    pub kind: VirtualTextureKind,
}

impl VirtualTexture {
    /// Base dimensions and subresource counts of this virtual texture.
    pub fn extent_and_subresources(&self) -> (u32, u32, u32, u32) {
        match &self.kind {
            VirtualTextureKind::Transient {
                desc,
                mip_levels,
                array_layers,
            } => (desc.width, desc.height, *mip_levels, *array_layers),
            VirtualTextureKind::Imported {
                width,
                height,
                mip_levels,
                array_layers,
                ..
            } => (*width, *height, *mip_levels, *array_layers),
        }
    }
}

/// Internal state tracking for a virtual buffer in the Render Graph.
#[derive(Clone, Debug)]
pub(crate) enum VirtualBufferKind {
    Transient(TransientBufferDesc),
    Imported {
        handle: BufferHandle,
        initial_access: BufferAccess,
        final_access: Option<BufferAccess>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct VirtualBuffer {
    pub kind: VirtualBufferKind,
}

#[derive(Clone, Debug)]
struct PooledTexture {
    desc: korppi_rhi::TextureDescriptor,
    handle: TextureHandle,
    frame_slot: usize,
    in_use: bool,
    last_access: Option<TextureAccess>,
}

#[derive(Clone, Debug)]
struct PooledBuffer {
    desc: korppi_rhi::BufferDescriptor,
    handle: BufferHandle,
    frame_slot: usize,
    in_use: bool,
    last_access: Option<BufferAccess>,
}

/// Pool of reusable physical textures and buffers allocated for transient render graph resources.
///
/// Tracks per-frame-in-flight slots and per-compilation acquisition so simultaneously live
/// transients never receive the same GPU handle (`RND-02`), and remembers each pooled resource's
/// last synchronization state across frames (`RND-05`).
pub struct TransientResourcePool {
    textures: Vec<PooledTexture>,
    buffers: Vec<PooledBuffer>,
    current_slot: usize,
}

impl Default for TransientResourcePool {
    fn default() -> Self {
        Self::new()
    }
}

impl TransientResourcePool {
    /// Creates a new empty transient resource pool.
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            buffers: Vec::new(),
            current_slot: 0,
        }
    }

    /// Selects the active frame-in-flight slot (`0..FRAMES_IN_FLIGHT`) and marks all pooled
    /// resources in that slot available for the upcoming frame's graph compilation (`RND-02`).
    pub fn begin_frame(&mut self, slot: FrameIndex) {
        self.current_slot = slot.index() % FRAMES_IN_FLIGHT;
        self.begin_compilation();
    }

    /// Resets per-compilation acquisition flags for the active frame slot.
    pub(crate) fn begin_compilation(&mut self) {
        for tex in &mut self.textures {
            if tex.frame_slot == self.current_slot {
                tex.in_use = false;
            }
        }
        for buf in &mut self.buffers {
            if buf.frame_slot == self.current_slot {
                buf.in_use = false;
            }
        }
    }

    /// Acquires or allocates a matching physical texture from the pool for the active frame slot.
    pub fn acquire_texture(
        &mut self,
        device: &dyn korppi_rhi::GraphicsDevice,
        desc: &TransientTextureDesc,
    ) -> Result<TextureHandle, korppi_rhi::RhiError> {
        self.acquire_texture_with_state(device, desc, 1, 1)
            .map(|(handle, _)| handle)
    }

    /// Acquires a physical texture along with its last recorded access state from a prior frame.
    pub(crate) fn acquire_texture_with_state(
        &mut self,
        device: &dyn korppi_rhi::GraphicsDevice,
        desc: &TransientTextureDesc,
        mip_levels: u32,
        array_layers: u32,
    ) -> Result<(TextureHandle, Option<TextureAccess>), korppi_rhi::RhiError> {
        let rhi_desc = korppi_rhi::TextureDescriptor {
            width: desc.width,
            height: desc.height,
            depth: 1,
            array_layers: array_layers.max(1),
            mip_levels: mip_levels.max(1),
            sample_count: desc.sample_count,
            format: desc.format,
            usage: desc.usage,
            dimension: desc.dimension,
            label: Some("transient_texture"),
        };

        for entry in &mut self.textures {
            if entry.frame_slot == self.current_slot
                && !entry.in_use
                && entry.desc.width == rhi_desc.width
                && entry.desc.height == rhi_desc.height
                && entry.desc.mip_levels == rhi_desc.mip_levels
                && entry.desc.array_layers == rhi_desc.array_layers
                && entry.desc.format == rhi_desc.format
                && entry.desc.usage == rhi_desc.usage
                && entry.desc.sample_count == rhi_desc.sample_count
                && entry.desc.dimension == rhi_desc.dimension
            {
                entry.in_use = true;
                return Ok((entry.handle, entry.last_access));
            }
        }

        let new_handle = device.create_texture(&rhi_desc, None)?;
        self.textures.push(PooledTexture {
            desc: rhi_desc,
            handle: new_handle,
            frame_slot: self.current_slot,
            in_use: true,
            last_access: None,
        });
        Ok((new_handle, None))
    }

    /// Records the final access state of a pooled texture at the end of a frame's graph.
    pub(crate) fn set_texture_last_access(&mut self, handle: TextureHandle, access: TextureAccess) {
        for entry in &mut self.textures {
            if entry.handle == handle {
                entry.last_access = Some(access);
                return;
            }
        }
    }

    /// Acquires or allocates a matching physical buffer from the pool for the active frame slot.
    pub fn acquire_buffer(
        &mut self,
        device: &dyn korppi_rhi::GraphicsDevice,
        desc: &TransientBufferDesc,
    ) -> Result<BufferHandle, korppi_rhi::RhiError> {
        self.acquire_buffer_with_state(device, desc)
            .map(|(handle, _)| handle)
    }

    /// Acquires a physical buffer along with its last recorded access state from a prior frame.
    pub(crate) fn acquire_buffer_with_state(
        &mut self,
        device: &dyn korppi_rhi::GraphicsDevice,
        desc: &TransientBufferDesc,
    ) -> Result<(BufferHandle, Option<BufferAccess>), korppi_rhi::RhiError> {
        let rhi_desc = korppi_rhi::BufferDescriptor {
            size: desc.size,
            usage: desc.usage,
            memory: korppi_rhi::MemoryUsage::GpuOnly,
            label: Some("transient_buffer"),
        };

        for entry in &mut self.buffers {
            if entry.frame_slot == self.current_slot
                && !entry.in_use
                && entry.desc.size >= rhi_desc.size
                && entry.desc.usage == rhi_desc.usage
            {
                entry.in_use = true;
                return Ok((entry.handle, entry.last_access));
            }
        }

        let new_handle = device.create_buffer(&rhi_desc, None)?;
        self.buffers.push(PooledBuffer {
            desc: rhi_desc,
            handle: new_handle,
            frame_slot: self.current_slot,
            in_use: true,
            last_access: None,
        });
        Ok((new_handle, None))
    }

    /// Records the final access state of a pooled buffer at the end of a frame's graph.
    pub(crate) fn set_buffer_last_access(&mut self, handle: BufferHandle, access: BufferAccess) {
        for entry in &mut self.buffers {
            if entry.handle == handle {
                entry.last_access = Some(access);
                return;
            }
        }
    }

    /// Destroys all pooled GPU resources across all frame slots.
    pub fn destroy(
        &mut self,
        device: &dyn korppi_rhi::GraphicsDevice,
    ) -> Result<(), korppi_rhi::RhiError> {
        for entry in self.textures.drain(..) {
            device.destroy_texture(entry.handle)?;
        }
        for entry in self.buffers.drain(..) {
            device.destroy_buffer(entry.handle)?;
        }
        Ok(())
    }
}
