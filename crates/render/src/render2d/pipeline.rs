//! Render Graph pass builder for batched 2D sprite rendering.

use crate::graph::{PassType, RenderGraph, RenderGraphTextureHandle};
use korppi_rhi::{LoadOp, StoreOp};

/// Helper pipeline to register 2D batched sprite passes onto a `RenderGraph`.
pub struct Render2dPipeline;

impl Render2dPipeline {
    /// Adds a 2D batched sprite rendering pass targeting `target`.
    pub fn build(
        graph: &mut RenderGraph,
        target: RenderGraphTextureHandle,
        width: u32,
        height: u32,
        clear_color: Option<[f32; 4]>,
    ) {
        let load_op = match clear_color {
            Some(_) => LoadOp::Clear,
            None => LoadOp::Load,
        };
        let clear = clear_color.unwrap_or([0.0, 0.0, 0.0, 0.0]);

        graph
            .add_pass("Batched2dSpritePass", PassType::Graphics)
            .write_color(target, 0, load_op, StoreOp::Store, clear)
            .mark_as_root()
            .record(move |_ctx, cmd| {
                cmd.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                cmd.set_scissor(0, 0, width, height);
            });
    }
}
