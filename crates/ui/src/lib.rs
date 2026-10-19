//! # Engine UI
//!
//! Godot-inspired Control node UI subsystem providing responsive layouts,
//! automated containers, theming, and event-driven widgets.

pub mod containers;
pub mod control;
pub mod input;
pub mod layout;
pub mod render;
pub mod theme;
pub mod widgets;

pub use containers::{
    fit_child_in_rect, resolve_container_layout, AspectRatioContainer, AspectStretchMode,
    BoxAlignment, BoxContainer, CenterContainer, DraggerVisibility, GridContainer, HBoxContainer,
    HSplitContainer, MarginContainer, Orientation, PanelContainer, ScrollContainer, ScrollMode,
    SplitContainer, VBoxContainer, VSplitContainer,
};
pub use control::{Control, FocusMode, GrowDirection, LayoutPreset, MouseFilter};
pub use input::{
    collect_front_to_back_order, hit_test_controls, FocusDirection, FocusManager, UiEvent,
    UiInputDispatcher, UiReceiver,
};
pub use layout::{
    compute_combined_minimum_size, is_effectively_visible, resolve_control_rect,
    resolve_control_tree, resolve_ui_layout_system, Rect2d, ResolvedRect, SizeFlags,
};
pub use render::{add_ui_clear_render_pass, add_ui_render_pass, UiCanvasRenderer, UiDrawCommand};
pub use theme::{
    get_theme_color, get_theme_constant, get_theme_font_size, get_theme_stylebox, AxisStretchMode,
    StyleBox, StyleBoxEmpty, StyleBoxFlat, StyleBoxTexture, Theme, ThemeOverride,
};
pub use widgets::{
    Button, ButtonState, CheckBox, CheckButton, ColorPicker, HorizontalAlignment, ItemList,
    ItemListItem, ItemListSelectMode, Label, LineEdit, MenuBar, MenuItem, OptionButton, PopupMenu,
    ProgressBar, ProgressFillMode, Range, Slider, SliderOrientation, TextEdit, Tree, TreeItem,
    VerticalAlignment,
};
