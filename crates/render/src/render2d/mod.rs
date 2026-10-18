//! 2D rendering pipeline: batched quads, texture atlasing, 9-patch scaling, bitmap fonts, and canvas modes.

pub mod atlas;
pub mod batch;
pub mod canvas;
pub mod debug;
pub mod gpu;
pub mod lighting;
pub mod pipeline;
pub mod text;

pub use atlas::{compute_9patch_quads, NinePatchMargins, NinePatchQuad, Rect2d, TextureAtlas2d};
pub use batch::{QuadBatch2d, QuadBatchDraw, QuadVertex2d};
pub use canvas::{compute_canvas_transform, CanvasScaleMode, CanvasTransform};
pub use debug::DebugDraw2d;
pub use gpu::QuadBatchRenderer;
pub use lighting::{Light2d, Occluder2d};
pub use pipeline::Render2dPipeline;
pub use text::BitmapFont2d;
