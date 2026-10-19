//! UI canvas hierarchy traversal, scissor stack management, and QuadBatch2d compilation.

use crate::containers::PanelContainer;
use crate::control::Control;
use crate::layout::hierarchy::{is_effectively_visible, ResolvedRect};
use crate::layout::Rect2d;
use crate::render::command::UiDrawCommand;
use crate::theme::{get_theme_color, get_theme_stylebox, StyleBox, ThemeOverride};
use crate::widgets::{
    Button, CheckBox, CheckButton, ColorPicker, ItemList, Label, LineEdit, MenuBar, OptionButton,
    PopupMenu, ProgressBar, Slider, TextEdit, Tree,
};
use korppi_ecs::{Entity, World};
use korppi_math::{Vec2, Vec4};
use korppi_render::render2d::{
    compute_9patch_quads, NinePatchMargins, QuadBatch2d, Rect2d as RenderRect2d,
};

const MAX_POOLED_UI_STRINGS: usize = 512;

thread_local! {
    static UI_TEXT_SCRATCH_POOL: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn recycle_commands_into_pool(commands: &mut Vec<UiDrawCommand>) {
    UI_TEXT_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        for cmd in commands.drain(..) {
            let mut s = match cmd {
                UiDrawCommand::Button { text, .. } | UiDrawCommand::Text { text, .. } => text,
                _ => continue,
            };
            if s.capacity() > 0 && pool.len() < MAX_POOLED_UI_STRINGS {
                s.clear();
                pool.push(s);
            }
        }
    });
}

fn take_pooled_string_with(min_cap: usize, fill: impl FnOnce(&mut String)) -> String {
    if min_cap == 0 {
        let mut s = String::new();
        fill(&mut s);
        return s;
    }
    let mut s = UI_TEXT_SCRATCH_POOL
        .with(|pool| {
            let mut pool = pool.borrow_mut();
            if let Some(pos) = pool.iter().rposition(|buf| buf.capacity() >= min_cap) {
                Some(pool.swap_remove(pos))
            } else {
                pool.pop()
            }
        })
        .unwrap_or_else(|| String::with_capacity(min_cap.max(128)));
    s.clear();
    fill(&mut s);
    s
}

fn take_pooled_string(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    take_pooled_string_with(text.len(), |s| s.push_str(text))
}

/// UI canvas renderer traversing resolved Control node trees and compiling into GPU quad batches.
#[derive(Clone, Debug, Default)]
pub struct UiCanvasRenderer;

impl UiCanvasRenderer {
    /// Creates a new `UiCanvasRenderer` and pre-warms the thread-local string scratch pool.
    pub fn new() -> Self {
        UI_TEXT_SCRATCH_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            if pool.capacity() < 64 {
                pool.reserve(64);
            }
            while pool.len() < 16 {
                pool.push(String::with_capacity(128));
            }
        });
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
        recycle_commands_into_pool(out);
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

        if roots_scratch.len() > 1 {
            roots_scratch.sort_unstable_by_key(|&e| {
                (world.get::<Control>(e).map_or(0, |c| c.z_index), e.index())
            });
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

        if !is_effectively_visible(world, entity) {
            return;
        }

        // Compute current effective scissor rectangle
        let current_scissor = if control.clip_contents {
            match parent_scissor {
                Some(p_sc) => match rect.intersection(&p_sc) {
                    Some(sc) => Some(sc),
                    None => return,
                },
                None => Some(rect),
            }
        } else {
            parent_scissor
        };

        // 1. StyleBox background (local ThemeOverride or PanelContainer theme cascade)
        let panel_style = world
            .get::<ThemeOverride>(entity)
            .and_then(|ov| ov.get_stylebox("panel").cloned())
            .or_else(|| {
                if world.has::<PanelContainer>(entity) {
                    get_theme_stylebox(world, entity, "Panel", "panel")
                } else {
                    None
                }
            });
        if let Some(style) = panel_style {
            match style {
                StyleBox::Flat(f) => {
                    out.push(UiDrawCommand::StyleBoxFlat {
                        rect,
                        style: f,
                        scissor: current_scissor,
                    });
                }
                StyleBox::Texture(t) => {
                    out.push(UiDrawCommand::StyleBoxTexture {
                        rect,
                        style: t,
                        scissor: current_scissor,
                    });
                }
                StyleBox::Empty(_) => {}
            }
        }

        // 2. Button widget
        if let Some(btn) = world.get::<Button>(entity) {
            out.push(UiDrawCommand::Button {
                rect,
                text: take_pooled_string(btn.text()),
                state: btn.state,
                scissor: current_scissor,
            });
        }

        // 3. Label widget
        if let Some(lbl) = world.get::<Label>(entity) {
            let color = get_theme_color(world, entity, "Label", "font_color")
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
            out.push(UiDrawCommand::Text {
                rect,
                text: take_pooled_string_with(lbl.text().len(), |s| lbl.write_display_text(s)),
                color,
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
                text: take_pooled_string(cb.text()),
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
                text: take_pooled_string_with(le.text().len(), |s| le.write_display_text(s)),
                color: [0.95, 0.95, 0.95, 1.0],
                alignment: Default::default(),
                scissor: current_scissor,
            });
        }

        // 8. CheckButton widget
        if let Some(cb) = world.get::<CheckButton>(entity) {
            let switch_w = 32.0;
            let switch_h = 18.0;
            let switch_rect = Rect2d::new(
                rect.x + (rect.width - switch_w).max(0.0),
                rect.y + (rect.height - switch_h) * 0.5,
                switch_w,
                switch_h,
            );
            let switch_color = if cb.is_checked() {
                [0.2, 0.65, 0.35, 1.0]
            } else {
                [0.2, 0.2, 0.24, 1.0]
            };
            out.push(UiDrawCommand::Quad {
                rect: switch_rect,
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ONE,
                color: switch_color,
                scissor: current_scissor,
            });
            let text_rect = Rect2d::new(
                rect.x,
                rect.y,
                (rect.width - switch_w - 6.0).max(0.0),
                rect.height,
            );
            out.push(UiDrawCommand::Text {
                rect: text_rect,
                text: take_pooled_string(cb.text()),
                color: [0.9, 0.9, 0.9, 1.0],
                alignment: Default::default(),
                scissor: current_scissor,
            });
        }

        // 9. TextEdit widget
        if let Some(te) = world.get::<TextEdit>(entity) {
            out.push(UiDrawCommand::Quad {
                rect,
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ONE,
                color: [0.12, 0.12, 0.14, 1.0],
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
                text: take_pooled_string(te.text()),
                color: [0.95, 0.95, 0.95, 1.0],
                alignment: Default::default(),
                scissor: current_scissor,
            });
        }

        // 10. Tree widget
        if let Some(tree) = world.get::<Tree>(entity) {
            out.push(UiDrawCommand::Quad {
                rect,
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ONE,
                color: [0.13, 0.14, 0.16, 1.0],
                scissor: current_scissor,
            });
            let mut row = 0usize;
            for &root_id in tree.root_items() {
                self.collect_tree_item_commands(
                    tree,
                    root_id,
                    0,
                    rect,
                    &mut row,
                    current_scissor,
                    out,
                );
            }
        }

        // 11. ItemList widget
        if let Some(list) = world.get::<ItemList>(entity) {
            out.push(UiDrawCommand::Quad {
                rect,
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ONE,
                color: [0.13, 0.14, 0.16, 1.0],
                scissor: current_scissor,
            });
            for i in 0..list.item_count() {
                if let Some(text) = list.get_item_text(i) {
                    let row_y = rect.y + (i as f32) * 24.0;
                    let row_rect =
                        Rect2d::new(rect.x + 6.0, row_y, (rect.width - 12.0).max(0.0), 24.0);
                    out.push(UiDrawCommand::Text {
                        rect: row_rect,
                        text: take_pooled_string(text),
                        color: [0.9, 0.9, 0.9, 1.0],
                        alignment: Default::default(),
                        scissor: current_scissor,
                    });
                }
            }
        }

        // 12. PopupMenu widget
        if let Some(popup) = world.get::<PopupMenu>(entity) {
            if popup.visible {
                out.push(UiDrawCommand::Quad {
                    rect,
                    uv_min: Vec2::ZERO,
                    uv_max: Vec2::ONE,
                    color: [0.16, 0.17, 0.20, 0.98],
                    scissor: current_scissor,
                });
                for i in 0..popup.item_count() {
                    if let Some(item) = popup.get_item(i) {
                        let row_y = rect.y + (i as f32) * 24.0;
                        if item.separator {
                            out.push(UiDrawCommand::Quad {
                                rect: Rect2d::new(
                                    rect.x + 4.0,
                                    row_y + 11.0,
                                    (rect.width - 8.0).max(0.0),
                                    2.0,
                                ),
                                uv_min: Vec2::ZERO,
                                uv_max: Vec2::ONE,
                                color: [0.3, 0.32, 0.36, 1.0],
                                scissor: current_scissor,
                            });
                        } else {
                            out.push(UiDrawCommand::Text {
                                rect: Rect2d::new(
                                    rect.x + 8.0,
                                    row_y,
                                    (rect.width - 16.0).max(0.0),
                                    24.0,
                                ),
                                text: take_pooled_string(&item.text),
                                color: if item.disabled {
                                    [0.5, 0.5, 0.5, 1.0]
                                } else {
                                    [0.92, 0.92, 0.92, 1.0]
                                },
                                alignment: Default::default(),
                                scissor: current_scissor,
                            });
                        }
                    }
                }
            }
        }

        // 13. MenuBar widget
        if let Some(bar) = world.get::<MenuBar>(entity) {
            out.push(UiDrawCommand::Quad {
                rect,
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ONE,
                color: [0.16, 0.17, 0.20, 1.0],
                scissor: current_scissor,
            });
            let mut x_cursor = rect.x + 8.0;
            for i in 0..bar.menu_count() {
                if let Some(title) = bar.get_menu_title(i) {
                    let w = (title.chars().count() as f32) * 8.0 + 16.0;
                    out.push(UiDrawCommand::Text {
                        rect: Rect2d::new(x_cursor, rect.y, w, rect.height),
                        text: take_pooled_string(title),
                        color: [0.92, 0.92, 0.92, 1.0],
                        alignment: Default::default(),
                        scissor: current_scissor,
                    });
                    x_cursor += w + 4.0;
                }
            }
        }

        // 14. OptionButton widget
        if let Some(opt) = world.get::<OptionButton>(entity) {
            out.push(UiDrawCommand::Button {
                rect,
                text: take_pooled_string(opt.selected_text().unwrap_or("")),
                state: crate::widgets::ButtonState::Normal,
                scissor: current_scissor,
            });
        }

        // 15. ColorPicker widget
        if let Some(picker) = world.get::<ColorPicker>(entity) {
            out.push(UiDrawCommand::Quad {
                rect,
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ONE,
                color: picker.color(),
                scissor: current_scissor,
            });
        }

        // Traverse children sorted by z_index
        let children_slice = world.children(entity);
        let needs_sort = children_slice.len() > 1
            && children_slice.windows(2).any(|w| {
                let z0 = world.get::<Control>(w[0]).map_or(0, |c| c.z_index);
                let z1 = world.get::<Control>(w[1]).map_or(0, |c| c.z_index);
                z0 > z1
            });
        if needs_sort {
            let mut children: korppi_core::SmallVec<[(usize, Entity); 16]> =
                children_slice.iter().copied().enumerate().collect();
            children.sort_unstable_by_key(|&(idx, c)| {
                (world.get::<Control>(c).map_or(0, |ctl| ctl.z_index), idx)
            });
            for (_, child) in children {
                self.collect_node_commands(world, child, current_scissor, out);
            }
        } else {
            for &child in children_slice {
                self.collect_node_commands(world, child, current_scissor, out);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_tree_item_commands(
        &self,
        tree: &Tree,
        item_id: usize,
        depth: usize,
        tree_rect: Rect2d,
        row: &mut usize,
        scissor: Option<Rect2d>,
        out: &mut Vec<UiDrawCommand>,
    ) {
        let Some(text) = tree.get_item_text(item_id) else {
            return;
        };
        let row_y = tree_rect.y + (*row as f32) * 24.0;
        *row += 1;

        if tree.selected_item() == Some(item_id) {
            out.push(UiDrawCommand::Quad {
                rect: Rect2d::new(tree_rect.x, row_y, tree_rect.width, 24.0),
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ONE,
                color: [0.22, 0.45, 0.78, 0.8],
                scissor,
            });
        }

        let indent = 6.0 + (depth as f32) * 16.0;
        out.push(UiDrawCommand::Text {
            rect: Rect2d::new(
                tree_rect.x + indent,
                row_y,
                (tree_rect.width - indent).max(0.0),
                24.0,
            ),
            text: take_pooled_string(text),
            color: [0.9, 0.9, 0.9, 1.0],
            alignment: Default::default(),
            scissor,
        });

        if !tree.is_collapsed(item_id) {
            for &child_id in tree.children_of(item_id) {
                self.collect_tree_item_commands(
                    tree,
                    child_id,
                    depth + 1,
                    tree_rect,
                    row,
                    scissor,
                    out,
                );
            }
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
        self.append_into_quad_batch(commands, batch);
    }

    /// Appends compiled `commands` into `batch` without clearing existing quads, reusing `batch`'s
    /// existing buffer capacities.
    pub fn append_into_quad_batch(&self, commands: &[UiDrawCommand], batch: &mut QuadBatch2d) {
        for cmd in commands {
            match cmd {
                UiDrawCommand::Quad {
                    rect,
                    uv_min,
                    uv_max,
                    color,
                    scissor,
                } => {
                    push_clipped_quad(batch, *rect, *uv_min, *uv_max, *color, *scissor);
                }
                UiDrawCommand::StyleBoxFlat {
                    rect,
                    style,
                    scissor,
                } => {
                    let draw_rect = style.get_draw_rect(*rect);
                    // Background quad
                    if style.draw_center {
                        push_clipped_quad(
                            batch,
                            draw_rect,
                            Vec2::ZERO,
                            Vec2::ONE,
                            style.bg_color,
                            *scissor,
                        );
                    }

                    // Border quads ([left, top, right, bottom])
                    let [b_left, b_top, b_right, b_bottom] = style.border_width;
                    if b_top > 0.0 {
                        let top_r = Rect2d::new(draw_rect.x, draw_rect.y, draw_rect.width, b_top);
                        push_clipped_quad(
                            batch,
                            top_r,
                            Vec2::ZERO,
                            Vec2::ONE,
                            style.border_color,
                            *scissor,
                        );
                    }
                    if b_bottom > 0.0 {
                        let bot_r = Rect2d::new(
                            draw_rect.x,
                            draw_rect.y + draw_rect.height - b_bottom,
                            draw_rect.width,
                            b_bottom,
                        );
                        push_clipped_quad(
                            batch,
                            bot_r,
                            Vec2::ZERO,
                            Vec2::ONE,
                            style.border_color,
                            *scissor,
                        );
                    }
                    if b_left > 0.0 {
                        let left_r =
                            Rect2d::new(draw_rect.x, draw_rect.y, b_left, draw_rect.height);
                        push_clipped_quad(
                            batch,
                            left_r,
                            Vec2::ZERO,
                            Vec2::ONE,
                            style.border_color,
                            *scissor,
                        );
                    }
                    if b_right > 0.0 {
                        let right_r = Rect2d::new(
                            draw_rect.x + draw_rect.width - b_right,
                            draw_rect.y,
                            b_right,
                            draw_rect.height,
                        );
                        push_clipped_quad(
                            batch,
                            right_r,
                            Vec2::ZERO,
                            Vec2::ONE,
                            style.border_color,
                            *scissor,
                        );
                    }
                }
                UiDrawCommand::StyleBoxTexture {
                    rect,
                    style,
                    scissor,
                } => {
                    let draw_rect = style.get_draw_rect(*rect);
                    let [m_left, m_top, m_right, m_bottom] = style.texture_margin;
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

                        for (idx, q) in sub_quads.into_iter().enumerate() {
                            if idx == 4 && !style.draw_center {
                                continue;
                            }
                            push_clipped_quad(
                                batch,
                                Rect2d::new(q.dest.x, q.dest.y, q.dest.width, q.dest.height),
                                Vec2::new(q.uv.x, q.uv.y),
                                Vec2::new(q.uv.z, q.uv.w),
                                style.modulate,
                                *scissor,
                            );
                        }
                    } else if style.draw_center {
                        push_clipped_quad(
                            batch,
                            draw_rect,
                            Vec2::ZERO,
                            Vec2::ONE,
                            style.modulate,
                            *scissor,
                        );
                    }
                }
                UiDrawCommand::Button {
                    rect,
                    state,
                    scissor,
                    ..
                } => {
                    let btn_color = match state {
                        crate::widgets::ButtonState::Normal => [0.22, 0.23, 0.27, 1.0],
                        crate::widgets::ButtonState::Hover => [0.28, 0.30, 0.36, 1.0],
                        crate::widgets::ButtonState::Pressed => [0.16, 0.17, 0.20, 1.0],
                        crate::widgets::ButtonState::Disabled => [0.12, 0.12, 0.14, 0.6],
                        crate::widgets::ButtonState::Focused => [0.24, 0.26, 0.32, 1.0],
                    };
                    push_clipped_quad(batch, *rect, Vec2::ZERO, Vec2::ONE, btn_color, *scissor);
                }
                UiDrawCommand::ProgressBar {
                    rect,
                    ratio,
                    fill_mode,
                    scissor,
                } => {
                    // Track background
                    push_clipped_quad(
                        batch,
                        *rect,
                        Vec2::ZERO,
                        Vec2::ONE,
                        [0.12, 0.12, 0.14, 1.0],
                        *scissor,
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
                    push_clipped_quad(
                        batch,
                        fill_rect,
                        Vec2::ZERO,
                        Vec2::ONE,
                        [0.2, 0.6, 1.0, 1.0],
                        *scissor,
                    );
                }
                UiDrawCommand::Slider {
                    rect,
                    ratio,
                    orientation,
                    scissor,
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
                    push_clipped_quad(
                        batch,
                        track_rect,
                        Vec2::ZERO,
                        Vec2::ONE,
                        [0.15, 0.15, 0.18, 1.0],
                        *scissor,
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
                    push_clipped_quad(
                        batch,
                        thumb_rect,
                        Vec2::ZERO,
                        Vec2::ONE,
                        [0.8, 0.8, 0.85, 1.0],
                        *scissor,
                    );
                }
                UiDrawCommand::Text {
                    rect,
                    color,
                    scissor,
                    ..
                } => {
                    // Text glyph rendering placeholder quad
                    push_clipped_quad(batch, *rect, Vec2::ZERO, Vec2::ONE, *color, *scissor);
                }
            }
        }
    }
}

fn push_clipped_quad(
    batch: &mut QuadBatch2d,
    rect: Rect2d,
    uv_min: Vec2,
    uv_max: Vec2,
    color: [f32; 4],
    scissor: Option<Rect2d>,
) {
    if let Some(sc) = scissor {
        let Some(clipped) = rect.intersection(&sc) else {
            return;
        };
        let (u0, v0, u1, v1) = if rect.width > 0.0 && rect.height > 0.0 {
            let tx0 = ((clipped.x - rect.x) / rect.width).clamp(0.0, 1.0);
            let ty0 = ((clipped.y - rect.y) / rect.height).clamp(0.0, 1.0);
            let tx1 = ((clipped.x + clipped.width - rect.x) / rect.width).clamp(0.0, 1.0);
            let ty1 = ((clipped.y + clipped.height - rect.y) / rect.height).clamp(0.0, 1.0);
            (
                uv_min.x + (uv_max.x - uv_min.x) * tx0,
                uv_min.y + (uv_max.y - uv_min.y) * ty0,
                uv_min.x + (uv_max.x - uv_min.x) * tx1,
                uv_min.y + (uv_max.y - uv_min.y) * ty1,
            )
        } else {
            (uv_min.x, uv_min.y, uv_max.x, uv_max.y)
        };
        batch.push_quad(
            clipped.min(),
            clipped.max(),
            Vec2::new(u0, v0),
            Vec2::new(u1, v1),
            color,
        );
    } else {
        batch.push_quad(rect.min(), rect.max(), uv_min, uv_max, color);
    }
}
