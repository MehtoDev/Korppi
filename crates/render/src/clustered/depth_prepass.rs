//! Declarative Depth Prepass construction and Render Graph integration.

use crate::clustered::reverse_z::ReverseZ;
use crate::graph::handle::RenderGraphTextureHandle;
use crate::graph::pass::PassType;
use crate::graph::render_graph::RenderGraph;
use crate::graph::resource::TransientTextureDesc;
use korppi_rhi::{LoadOp, SampleCount, StoreOp, TextureDimension, TextureFormat, TextureUsage};

/// Configuration options for configuring a Depth Prepass.
#[derive(Clone, Debug)]
pub struct DepthPrepassConfig {
    /// Render target width.
    pub width: u32,
    /// Render target height.
    pub height: u32,
    /// Texture format (defaults to `Depth32Float`).
    pub format: TextureFormat,
    /// Hardware sample count for MSAA.
    pub sample_count: SampleCount,
}

impl Default for DepthPrepassConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            format: TextureFormat::Depth32Float,
            sample_count: SampleCount::Count1,
        }
    }
}

/// Adds a standard Depth Prepass to the provided Render Graph.
///
/// Creates a transient depth texture, clears it to `ReverseZ::CLEAR_DEPTH` (0.0),
/// configures the viewport and scissor, and returns the virtual depth texture handle.
pub fn add_depth_prepass(
    graph: &mut RenderGraph,
    config: &DepthPrepassConfig,
) -> RenderGraphTextureHandle {
    let depth_desc = TransientTextureDesc {
        name: "DepthPrepassTarget".to_string(),
        width: config.width,
        height: config.height,
        format: config.format,
        usage: TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
        dimension: TextureDimension::D2,
        sample_count: config.sample_count,
    };
    let depth_handle = graph.create_transient_texture(depth_desc);

    let (w, h) = (config.width as f32, config.height as f32);
    graph
        .add_pass("DepthPrepass", PassType::Graphics)
        .write_depth(
            depth_handle,
            LoadOp::Clear,
            StoreOp::Store,
            ReverseZ::CLEAR_DEPTH,
        )
        .record(move |_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, w, h, 0.0, 1.0);
            cmd.set_scissor(0, 0, w as u32, h as u32);
        });

    depth_handle
}
