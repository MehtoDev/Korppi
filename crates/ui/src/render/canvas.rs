//! UI canvas hierarchy traversal, scissor stack management, and QuadBatch2d compilation.

use crate::control::Control;
use crate::layout::hierarchy::ResolvedRect;
use crate::layout::Rect2d;
use crate::render::command::UiDrawCommand;
use crate::theme::{StyleBox, ThemeOverride};
use crate::widgets::{Button, CheckBox, Label, LineEdit, ProgressBar, Slider};
use korppi_ecs::{Entity, World};
use korppi_math::{Vec2, Vec4};
use korppi_render::render2d::{
    compute_9patch_quads, NinePatchMargins, QuadBatch2d, Rect2d as RenderRect2d,
};

/// UI canvas renderer traversing resolved Control node trees and compiling into GPU quad batches.
#[derive(Clone, Debug, Default)]
pub struct UiCanvasRenderer;

impl UiCanvasRenderer {
    /// Creates a new `UiCanvasRenderer`.
    pub fn new() -> Self {
        Self
    }

    /// Traverses the ECS hierarchy in scene-graph rendering order and collects intermediate draw commands.
    pub fn collect_draw_commands(&self, world: &World) -> Vec<UiDrawCommand> {
        let mut roots = Vec::new();
        let mut commands = Vec::new();
        self.collect_draw_commands_into(world, &mut roots, &mut commands);
        commands
    }

    /// Traverses the ECS hierarchy in scene-graph rendering order and appends intermediate draw
    /// commands into `out`, reusing `roots_scratch` and `out` capacities without heap allocation.
    pub fn collect_draw_commands_into(
        &self,
        world: &World,
        roots_scratch: &mut Vec<Entity>,
        out: &mut Vec<UiDrawCommand>,
    ) {
        roots_scratch.clear();
        out.clear();
        for entity in world.entities() {
            if world.has::<Control>(entity) {
                let is_root = if let Some(parent) = world.parent_of(entity) {
                    !world.has::<Control>(parent)
                } else {
                    true
                };
                if is_root {
                    roots_scratch.push(entity);
                }
            }
        }

        for &root in roots_scratch.iter() {
            self.collect_node_commands(world, root, None, out);
        }
    }

    fn collect_node_commands(
        &self,
        world: &World,
        entity: Entity,
        parent_scissor: Option<Rect2d>,
        out: &mut Vec<UiDrawCommand>,
    ) {
        let (control, rect) = match (
            world.get::<Control>(entity),
            world.get::<ResolvedRect>(entity),
        ) {
            (Some(c), Some(r)) => (c, r.global),
            _ => return,
        };

        if !control.visible {
            return;
        }

        // Compute current effective scissor rectangle
        let current_scissor = if control.clip_contents {
            match parent_scissor {
                Some(p_sc) => rect.intersection(&p_sc).or(Some(rect)),
                None => Some(rect),
            }
        } else {
            parent_scissor
        };

        // 1. StyleBox background
        if let Some(th_override) = world.get::<ThemeOverride>(entity) {
            if let Some(style) = th_override.styleboxes.get("panel") {
                match style {
                    StyleBox::Flat(f) => {
                        out.push(UiDrawCommand::StyleBoxFlat {
                            rect,
                            style: f.clone(),
                            scissor: current_scissor,
                        });
                    }
                    StyleBox::Texture(t) => {
                        out.push(UiDrawCommand::StyleBoxTexture {
                            rect,
                            style: t.clone(),
                            scissor: current_scissor,
                        });
                    }
                    StyleBox::Empty(_) => {}
                }
            }
        }

        // 2. Button widget
        if let Some(btn) = world.get::<Button>(entity) {
            out.push(UiDrawCommand::Button {
                rect,
                text: btn.text().to_string(),
                state: btn.state,
                scissor: current_scissor,
            });
        }

        // 3. Label widget
        if let Some(lbl) = world.get::<Label>(entity) {
            out.push(UiDrawCommand::Text {
                rect,
                text: lbl.display_text(),
                color: [1.0, 1.0, 1.0, 1.0],
                alignment: (lbl.horizontal_alignment, lbl.vertical_alignment),
                scissor: current_scissor,
            });
        }

        // 4. ProgressBar widget
        if let Some(pbar) = world.get::<ProgressBar>(entity) {
            out.push(UiDrawCommand::ProgressBar {
                rect,
                ratio: pbar.ratio() as f32,
                fill_mode: pbar.fill_mode,
                scissor: current_scissor,
            });
        }

        // 5. Slider widget
        if let Some(slider) = world.get::<Slider>(entity) {
            out.push(UiDrawCommand::Slider {
                rect,
                ratio: slider.range.ratio() as f32,
                orientation: slider.orientation,
                scissor: current_scissor,
            });
        }

        // 6. CheckBox widget
        if let Some(cb) = world.get::<CheckBox>(entity) {
            let box_rect = Rect2d::new(rect.x, rect.y + (rect.height - 18.0) * 0.5, 18.0, 18.0);
            let check_color = if cb.is_checked() {
                [0.2, 0.6, 1.0, 1.0]
            } else {
                [0.15, 0.15, 0.15, 1.0]
            };
            out.push(UiDrawCommand::Quad {
                rect: box_rect,
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ONE,
                color: check_color,
                scissor: current_scissor,
            });
            let text_rect = Rect2d::new(
                rect.x + 24.0,
                rect.y,
                (rect.width - 24.0).max(0.0),
                rect.height,
            );
            out.push(UiDrawCommand::Text {
                rect: text_rect,
                text: cb.text().to_string(),
                color: [0.9, 0.9, 0.9, 1.0],
                alignment: Default::default(),
                scissor: current_scissor,
            });
        }

        // 7. LineEdit widget
        if let Some(le) = world.get::<LineEdit>(entity) {
            let bg_color = [0.12, 0.12, 0.14, 1.0];
            out.push(UiDrawCommand::Quad {
                rect,
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ONE,
                color: bg_color,
                scissor: current_scissor,
            });
            let text_rect = Rect2d::new(
                rect.x + 6.0,
                rect.y + 4.0,
                (rect.width - 12.0).max(0.0),
                (rect.height - 8.0).max(0.0),
            );
            out.push(UiDrawCommand::Text {
                rect: text_rect,
                text: le.display_text(),
                color: [0.95, 0.95, 0.95, 1.0],
                alignment: Default::default(),
                scissor: current_scissor,
            });
        }

        // Traverse children
        for &child in world.children(entity) {
            self.collect_node_commands(world, child, current_scissor, out);
        }
    }

    /// Compiles intermediate draw commands into batched 2D quads (`QuadBatch2d`).
    pub fn compile_to_quad_batch(
        &self,
        commands: &[UiDrawCommand],
        max_quads: usize,
    ) -> QuadBatch2d {
        let mut batch = QuadBatch2d::new(max_quads);
        self.compile_into_quad_batch(commands, &mut batch);
        batch
    }

    /// Clears `batch` and compiles `commands` into it, reusing `batch`'s existing buffer capacities.
    pub fn compile_into_quad_batch(&self, commands: &[UiDrawCommand], batch: &mut QuadBatch2d) {
        batch.clear();

        for cmd in commands {
            match cmd {
                UiDrawCommand::Quad {
                    rect,
                    uv_min,
                    uv_max,
                    color,
                    ..
                } => {
                    batch.push_quad(rect.min(), rect.max(), *uv_min, *uv_max, *color);
                }
                UiDrawCommand::StyleBoxFlat { rect, style, .. } => {
                    let draw_rect = style.get_draw_rect(*rect);
                    // Background quad
                    batch.push_quad(
                        draw_rect.min(),
                        draw_rect.max(),
                        Vec2::ZERO,
                        Vec2::ONE,
                        style.bg_color,
                    );

                    // Border quads (top, bottom, left, right)
                    let [b_top, b_right, b_bottom, b_left] = style.border_width;
                    if b_top > 0.0 {
                        let top_r = Rect2d::new(draw_rect.x, draw_rect.y, draw_rect.width, b_top);
                        batch.push_quad(
                            top_r.min(),
                            top_r.max(),
                            Vec2::ZERO,
                            Vec2::ONE,
                            style.border_color,
                        );
                    }
                    if b_bottom > 0.0 {
                        let bot_r = Rect2d::new(
                            draw_rect.x,
                            draw_rect.y + draw_rect.height - b_bottom,
                            draw_rect.width,
                            b_bottom,
                        );
                        batch.push_quad(
                            bot_r.min(),
                            bot_r.max(),
                            Vec2::ZERO,
                            Vec2::ONE,
                            style.border_color,
                        );
                    }
                    if b_left > 0.0 {
                        let left_r =
                            Rect2d::new(draw_rect.x, draw_rect.y, b_left, draw_rect.height);
                        batch.push_quad(
                            left_r.min(),
                            left_r.max(),
                            Vec2::ZERO,
                            Vec2::ONE,
                            style.border_color,
                        );
                    }
                    if b_right > 0.0 {
                        let right_r = Rect2d::new(
                            draw_rect.x + draw_rect.width - b_right,
                            draw_rect.y,
                            b_right,
                            draw_rect.height,
                        );
                        batch.push_quad(
                            right_r.min(),
                            right_r.max(),
                            Vec2::ZERO,
                            Vec2::ONE,
                            style.border_color,
                        );
                    }
                }
                UiDrawCommand::StyleBoxTexture { rect, style, .. } => {
                    let draw_rect = style.get_draw_rect(*rect);
                    let [m_top, m_right, m_bottom, m_left] = style.texture_margin;
                    if m_top > 0.0 || m_right > 0.0 || m_bottom > 0.0 || m_left > 0.0 {
                        let sub_quads = compute_9patch_quads(
                            RenderRect2d {
                                x: style.region_rect.x,
                                y: style.region_rect.y,
                                width: style.region_rect.width,
                                height: style.region_rect.height,
                            },
                            Vec4::new(0.0, 0.0, 1.0, 1.0),
                            NinePatchMargins {
                                left: m_left,
                                right: m_right,
                                top: m_top,
                                bottom: m_bottom,
                            },
                            RenderRect2d {
                                x: draw_rect.x,
                                y: draw_rect.y,
                                width: draw_rect.width,
                                height: draw_rect.height,
                            },
                        );

                        for q in sub_quads {
                            batch.push_quad(
                                Vec2::new(q.dest.x, q.dest.y),
                                Vec2::new(q.dest.x + q.dest.width, q.dest.y + q.dest.height),
                                Vec2::new(q.uv.x, q.uv.y),
                                Vec2::new(q.uv.z, q.uv.w),
                                [1.0, 1.0, 1.0, 1.0],
                            );
                        }
                    } else {
                        batch.push_quad(
                            draw_rect.min(),
                            draw_rect.max(),
                            Vec2::ZERO,
                            Vec2::ONE,
                            [1.0, 1.0, 1.0, 1.0],
                        );
                    }
                }
                UiDrawCommand::Button { rect, state, .. } => {
                    let btn_color = match state {
                        crate::widgets::ButtonState::Normal => [0.22, 0.23, 0.27, 1.0],
                        crate::widgets::ButtonState::Hover => [0.28, 0.30, 0.36, 1.0],
                        crate::widgets::ButtonState::Pressed => [0.16, 0.17, 0.20, 1.0],
                        crate::widgets::ButtonState::Disabled => [0.12, 0.12, 0.14, 0.6],
                        crate::widgets::ButtonState::Focused => [0.24, 0.26, 0.32, 1.0],
                    };
                    batch.push_quad(rect.min(), rect.max(), Vec2::ZERO, Vec2::ONE, btn_color);
                }
                UiDrawCommand::ProgressBar {
                    rect,
                    ratio,
                    fill_mode,
                    ..
                } => {
                    // Track background
                    batch.push_quad(
                        rect.min(),
                        rect.max(),
                        Vec2::ZERO,
                        Vec2::ONE,
                        [0.12, 0.12, 0.14, 1.0],
                    );

                    // Fill bar
                    let r = ratio.clamp(0.0, 1.0);
                    let fill_rect = match fill_mode {
                        crate::widgets::ProgressFillMode::LeftToRight => {
                            Rect2d::new(rect.x, rect.y, rect.width * r, rect.height)
                        }
                        crate::widgets::ProgressFillMode::RightToLeft => Rect2d::new(
                            rect.x + rect.width * (1.0 - r),
                            rect.y,
                            rect.width * r,
                            rect.height,
                        ),
                        crate::widgets::ProgressFillMode::BottomToTop => Rect2d::new(
                            rect.x,
                            rect.y + rect.height * (1.0 - r),
                            rect.width,
                            rect.height * r,
                        ),
                        crate::widgets::ProgressFillMode::TopToBottom => {
                            Rect2d::new(rect.x, rect.y, rect.width, rect.height * r)
                        }
                    };
                    batch.push_quad(
                        fill_rect.min(),
                        fill_rect.max(),
                        Vec2::ZERO,
                        Vec2::ONE,
                        [0.2, 0.6, 1.0, 1.0],
                    );
                }
                UiDrawCommand::Slider {
                    rect,
                    ratio,
                    orientation,
                    ..
                } => {
                    // Slider track
                    let r = ratio.clamp(0.0, 1.0);
                    let track_rect = match orientation {
                        crate::widgets::SliderOrientation::Horizontal => {
                            Rect2d::new(rect.x, rect.y + (rect.height - 4.0) * 0.5, rect.width, 4.0)
                        }
                        crate::widgets::SliderOrientation::Vertical => {
                            Rect2d::new(rect.x + (rect.width - 4.0) * 0.5, rect.y, 4.0, rect.height)
                        }
                    };
                    batch.push_quad(
                        track_rect.min(),
                        track_rect.max(),
                        Vec2::ZERO,
                        Vec2::ONE,
                        [0.15, 0.15, 0.18, 1.0],
                    );

                    // Thumb grabber
                    let thumb_rect = match orientation {
                        crate::widgets::SliderOrientation::Horizontal => {
                            let cx = rect.x + rect.width * r;
                            Rect2d::new(cx - 6.0, rect.y + (rect.height - 16.0) * 0.5, 12.0, 16.0)
                        }
                        crate::widgets::SliderOrientation::Vertical => {
                            let cy = rect.y + rect.height * (1.0 - r);
                            Rect2d::new(rect.x + (rect.width - 16.0) * 0.5, cy - 6.0, 16.0, 12.0)
                        }
                    };
                    batch.push_quad(
                        thumb_rect.min(),
                        thumb_rect.max(),
                        Vec2::ZERO,
                        Vec2::ONE,
                        [0.8, 0.8, 0.85, 1.0],
                    );
                }
                UiDrawCommand::Text { rect, color, .. } => {
                    // Text glyph rendering placeholder quad
                    batch.push_quad(rect.min(), rect.max(), Vec2::ZERO, Vec2::ONE, *color);
                }
            }
        }
    }
}
