//! Declarative Render Graph builder for post-processing and HDR composite passes (`RND-10`, `RND-18`).

use crate::graph::{
    PassType, RenderGraph, RenderGraphTextureHandle, TextureAccess, TransientTextureDesc,
};
use crate::postprocess::bloom::BloomConfig;
use crate::postprocess::ssao::SsaoConfig;
use crate::postprocess::tonemap::{evaluate_tonemapping, TonemapConfig};
use korppi_math::Vec3;
use korppi_rhi::{
    AccessFlags, ImageLayout, LoadOp, PipelineStageFlags, StoreOp, TextureFormat, TextureUsage,
};

const BLOOM_MIP_NAMES: [&str; 6] = [
    "BloomMip_0",
    "BloomMip_1",
    "BloomMip_2",
    "BloomMip_3",
    "BloomMip_4",
    "BloomMip_5",
];

const BLOOM_DOWN_NAMES: [&str; 6] = [
    "BloomDownsamplePass_0",
    "BloomDownsamplePass_1",
    "BloomDownsamplePass_2",
    "BloomDownsamplePass_3",
    "BloomDownsamplePass_4",
    "BloomDownsamplePass_5",
];

const BLOOM_UP_NAMES: [&str; 5] = [
    "BloomUpsamplePass_0",
    "BloomUpsamplePass_1",
    "BloomUpsamplePass_2",
    "BloomUpsamplePass_3",
    "BloomUpsamplePass_4",
];

/// High-level builder to declare post-processing passes (SSAO, Bloom, Tonemapping) onto a [`RenderGraph`].
#[derive(Clone, Debug, Default)]
pub struct PostProcessPipeline {
    /// Tonemapping configuration.
    pub tonemap: TonemapConfig,
    /// Optional bloom pyramid configuration.
    pub bloom: Option<BloomConfig>,
    /// Optional SSAO configuration.
    pub ssao: Option<SsaoConfig>,
}

impl PostProcessPipeline {
    /// Creates a configured [`PostProcessPipeline`].
    pub const fn new(
        tonemap: TonemapConfig,
        bloom: Option<BloomConfig>,
        ssao: Option<SsaoConfig>,
    ) -> Self {
        Self {
            tonemap,
            bloom,
            ssao,
        }
    }

    /// Appends this pipeline's configured passes onto `graph`.
    pub fn setup_graph(
        &self,
        graph: &mut RenderGraph,
        hdr_color: RenderGraphTextureHandle,
        depth: Option<RenderGraphTextureHandle>,
        ldr_target: RenderGraphTextureHandle,
    ) {
        Self::build(
            graph,
            hdr_color,
            depth,
            ldr_target,
            self.tonemap,
            self.bloom,
            self.ssao,
        );
    }

    /// Appends the complete HDR post-processing chain (SSAO -> Bloom -> Tonemap) onto a `RenderGraph` (`RND-18`).
    pub fn build(
        graph: &mut RenderGraph,
        hdr_color: RenderGraphTextureHandle,
        depth: Option<RenderGraphTextureHandle>,
        ldr_target: RenderGraphTextureHandle,
        tonemap_config: TonemapConfig,
        bloom_config: Option<BloomConfig>,
        ssao_config: Option<SsaoConfig>,
    ) {
        let (width, height) = graph
            .texture_dimensions(ldr_target)
            .or_else(|| graph.texture_dimensions(hdr_color))
            .unwrap_or((1920, 1080));
        let width = width.max(1);
        let height = height.max(1);

        let shader_read = TextureAccess {
            stage: PipelineStageFlags::FRAGMENT_SHADER,
            access: AccessFlags::SHADER_READ,
            layout: ImageLayout::ShaderReadOnly,
        };

        // 1. Optional SSAO generation + bilateral blur passes if depth buffer and SsaoConfig are provided
        let ssao_out = if let (Some(depth_handle), Some(ssao_cfg)) = (depth, ssao_config) {
            let half_w = (width / 2).max(1);
            let half_h = (height / 2).max(1);
            let ao_val = (1.0 - ssao_cfg.intensity * 0.05).clamp(0.0, 1.0);

            let ssao_raw = graph.create_transient_texture(TransientTextureDesc::new(
                "SsaoRaw",
                half_w,
                half_h,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
            ));
            let ssao_blur = graph.create_transient_texture(TransientTextureDesc::new(
                "SsaoBlurred",
                width,
                height,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
            ));

            graph
                .add_pass("SsaoGenerationPass", PassType::Graphics)
                .read_texture(depth_handle, shader_read)
                .write_color(
                    ssao_raw,
                    0,
                    LoadOp::Clear,
                    StoreOp::Store,
                    [ao_val, ao_val, ao_val, 1.0],
                )
                .record(move |_ctx, cmd| {
                    cmd.set_viewport(0.0, 0.0, half_w as f32, half_h as f32, 0.0, 1.0);
                    cmd.set_scissor(0, 0, half_w, half_h);
                });

            graph
                .add_pass("SsaoBlurPass", PassType::Graphics)
                .read_texture(ssao_raw, shader_read)
                .read_texture(depth_handle, shader_read)
                .write_color(
                    ssao_blur,
                    0,
                    LoadOp::Clear,
                    StoreOp::Store,
                    [ao_val, ao_val, ao_val, 1.0],
                )
                .record(move |_ctx, cmd| {
                    cmd.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                    cmd.set_scissor(0, 0, width, height);
                });

            Some((ssao_blur, ao_val))
        } else {
            None
        };

        // 2. Optional progressive Bloom downsample & upsample mip chain
        let bloom_out = if let Some(bloom_cfg) = bloom_config {
            let mip_count = (bloom_cfg.mip_levels as usize).clamp(1, 6);
            let mut mips = [hdr_color; 6];
            let mut mip_extents = [(1u32, 1u32); 6];

            for i in 0..mip_count {
                let mw = (width >> (i + 1)).max(1);
                let mh = (height >> (i + 1)).max(1);
                mip_extents[i] = (mw, mh);
                mips[i] = graph.create_transient_texture(TransientTextureDesc::new(
                    BLOOM_MIP_NAMES[i],
                    mw,
                    mh,
                    TextureFormat::Rgba16Float,
                    TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
                ));
            }

            for i in 0..mip_count {
                let src = if i == 0 { hdr_color } else { mips[i - 1] };
                let (mw, mh) = mip_extents[i];
                graph
                    .add_pass(BLOOM_DOWN_NAMES[i], PassType::Graphics)
                    .read_texture(src, shader_read)
                    .write_color(
                        mips[i],
                        0,
                        LoadOp::Clear,
                        StoreOp::Store,
                        [0.0, 0.0, 0.0, 1.0],
                    )
                    .record(move |_ctx, cmd| {
                        cmd.set_viewport(0.0, 0.0, mw as f32, mh as f32, 0.0, 1.0);
                        cmd.set_scissor(0, 0, mw, mh);
                    });
            }

            for (step, &pass_name) in BLOOM_UP_NAMES
                .iter()
                .enumerate()
                .take(mip_count.saturating_sub(1))
            {
                let src_mip = mip_count - 1 - step;
                let dst_mip = src_mip - 1;
                let (dw, dh) = mip_extents[dst_mip];
                graph
                    .add_pass(pass_name, PassType::Graphics)
                    .read_texture(mips[src_mip], shader_read)
                    .write_color(
                        mips[dst_mip],
                        0,
                        LoadOp::Load,
                        StoreOp::Store,
                        [0.0, 0.0, 0.0, 1.0],
                    )
                    .record(move |_ctx, cmd| {
                        cmd.set_viewport(0.0, 0.0, dw as f32, dh as f32, 0.0, 1.0);
                        cmd.set_scissor(0, 0, dw, dh);
                    });
            }

            Some((mips[0], bloom_cfg.intensity.max(0.0)))
        } else {
            None
        };

        // 3. Tonemapping and final LDR composite pass
        let ao_factor = ssao_out.map(|(_, ao)| ao).unwrap_or(1.0);
        let bloom_contrib = bloom_out.map(|(_, b)| b).unwrap_or(0.0);
        let ref_hdr = Vec3::splat(0.18 * ao_factor + bloom_contrib);
        let tonemapped = evaluate_tonemapping(ref_hdr, &tonemap_config);
        let composite_clear = [tonemapped.x, tonemapped.y, tonemapped.z, 1.0];

        let mut composite = graph
            .add_pass("PostProcessCompositePass", PassType::Graphics)
            .read_texture(hdr_color, shader_read);
        if let Some((ssao_tex, _)) = ssao_out {
            composite = composite.read_texture(ssao_tex, shader_read);
        }
        if let Some((bloom_tex, _)) = bloom_out {
            composite = composite.read_texture(bloom_tex, shader_read);
        }
        composite
            .write_color(
                ldr_target,
                0,
                LoadOp::Clear,
                StoreOp::Store,
                composite_clear,
            )
            .mark_as_root()
            .record(move |_ctx, cmd| {
                cmd.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                cmd.set_scissor(0, 0, width, height);
            });
    }
}
