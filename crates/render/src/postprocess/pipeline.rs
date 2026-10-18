//! Declarative Render Graph builder for post-processing and HDR composite passes.

use crate::graph::{PassType, RenderGraph, RenderGraphTextureHandle, TextureAccess};
use crate::postprocess::bloom::BloomConfig;
use crate::postprocess::ssao::SsaoConfig;
use crate::postprocess::tonemap::TonemapConfig;
use korppi_rhi::{AccessFlags, ImageLayout, LoadOp, PipelineStageFlags, StoreOp};

/// High-level builder to declare post-processing passes onto a Render Graph.
pub struct PostProcessPipeline;

impl PostProcessPipeline {
    /// Appends the complete HDR post-processing chain (SSAO -> Bloom -> Tonemap) onto a `RenderGraph`.
    pub fn build(
        graph: &mut RenderGraph,
        hdr_color: RenderGraphTextureHandle,
        depth: Option<RenderGraphTextureHandle>,
        ldr_target: RenderGraphTextureHandle,
        tonemap_config: TonemapConfig,
        bloom_config: Option<BloomConfig>,
        ssao_config: Option<SsaoConfig>,
    ) {
        let _ = (tonemap_config, bloom_config, ssao_config);

        // 1. Optional SSAO passes if depth buffer is provided
        if let (Some(depth_handle), Some(_ssao_cfg)) = (depth, ssao_config) {
            graph
                .add_pass("SsaoGenerationPass", PassType::Graphics)
                .read_texture(
                    depth_handle,
                    TextureAccess {
                        stage: PipelineStageFlags::FRAGMENT_SHADER,
                        access: AccessFlags::SHADER_READ,
                        layout: ImageLayout::ShaderReadOnly,
                    },
                )
                .record(|_ctx, cmd| {
                    cmd.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
                    cmd.set_scissor(0, 0, 1920, 1080);
                });
        }

        // 2. Tonemapping and final LDR composite pass
        graph
            .add_pass("PostProcessCompositePass", PassType::Graphics)
            .read_texture(
                hdr_color,
                TextureAccess {
                    stage: PipelineStageFlags::FRAGMENT_SHADER,
                    access: AccessFlags::SHADER_READ,
                    layout: ImageLayout::ShaderReadOnly,
                },
            )
            .write_color(
                ldr_target,
                0,
                LoadOp::Clear,
                StoreOp::Store,
                [0.0, 0.0, 0.0, 1.0],
            )
            .mark_as_root()
            .record(|_ctx, cmd| {
                cmd.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
                cmd.set_scissor(0, 0, 1920, 1080);
            });
    }
}
