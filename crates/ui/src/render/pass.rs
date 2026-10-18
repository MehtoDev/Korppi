//! Render Graph integration pass for UI canvas rendering.

use korppi_render::graph::{PassType, RenderGraph, RenderGraphTextureHandle};
use korppi_render::render2d::QuadBatch2d;
use korppi_rhi::{LoadOp, StoreOp};

/// Schedules a 2D UI canvas pass in the Render Graph targeting the specified virtual render target.
pub fn add_ui_render_pass(
    graph: &mut RenderGraph,
    color_target: RenderGraphTextureHandle,
    batch: QuadBatch2d,
) {
    graph
        .add_pass("ui_canvas_pass", PassType::Graphics)
        .write_color(
            color_target,
            0,
            LoadOp::Load,
            StoreOp::Store,
            [0.0, 0.0, 0.0, 0.0],
        )
        .mark_as_root()
        .record(move |_ctx, _recorder| {
            // In full GPU execution, binds UI pipeline and issues draw calls for batch.quad_count()
            let _ = batch.quad_count();
        });
}

/// Schedules a root 2D UI canvas pass in `graph` that clears `color_target` to `clear_color`
/// and prepares it for execution via [`RenderGraph::execute_with`].
pub fn add_ui_clear_render_pass(
    graph: &mut RenderGraph,
    color_target: RenderGraphTextureHandle,
    clear_color: [f32; 4],
) {
    graph
        .add_pass("ui_canvas_pass", PassType::Graphics)
        .write_color(color_target, 0, LoadOp::Clear, StoreOp::Store, clear_color)
        .mark_as_root();
}
