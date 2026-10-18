//! Comprehensive unit tests for Godot-inspired Core Widget Library:
//! Label, Button, LineEdit, TextEdit, Slider, ProgressBar, CheckBox, CheckButton,
//! Tree, ItemList, PopupMenu, MenuBar, OptionButton, and ColorPicker.

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_input::mouse::MouseButton;
use korppi_math::Vec2;
use korppi_ui::control::{Control, LayoutPreset};
use korppi_ui::input::{UiEvent, UiInputDispatcher, UiReceiver};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d};
use korppi_ui::widgets::{
    Button, ButtonState, CheckBox, CheckButton, ColorPicker, HorizontalAlignment, ItemList,
    ItemListSelectMode, Label, LineEdit, MenuBar, OptionButton, PopupMenu, ProgressBar,
    ProgressFillMode, Range, Slider, SliderOrientation, TextEdit, Tree, VerticalAlignment,
};

#[test]
fn test_label_text_metrics_and_formatting() {
    let mut label = Label::new("Hello World");
    assert_eq!(label.text(), "Hello World");
    assert_eq!(label.line_count(), 1);

    // Multi-line text
    label.set_text("Line 1\nLine 2\nLine 3");
    assert_eq!(label.line_count(), 3);
    assert_eq!(label.get_line(1), Some("Line 2"));

    // Uppercase transformation
    label.uppercase = true;
    assert_eq!(label.display_text(), "LINE 1\nLINE 2\nLINE 3");

    // Visible characters (typewriter effect)
    label.uppercase = false;
    label.set_text("Dialogue line");
    label.visible_characters = Some(8);
    assert_eq!(label.display_text(), "Dialogue");

    // Alignments
    label.horizontal_alignment = HorizontalAlignment::Center;
    label.vertical_alignment = VerticalAlignment::Center;
    assert_eq!(label.horizontal_alignment, HorizontalAlignment::Center);
    assert_eq!(label.vertical_alignment, VerticalAlignment::Center);
}

#[test]
fn test_button_states_and_toggle_mode() {
    let mut btn = Button::new("Click Me");
    assert_eq!(btn.text(), "Click Me");
    assert_eq!(btn.state, ButtonState::Normal);
    assert!(!btn.is_pressed());

    // Hover state
    btn.handle_event(&UiEvent::MouseEnter);
    assert_eq!(btn.state, ButtonState::Hover);

    // Press down
    btn.handle_event(&UiEvent::MouseButton {
        position: Vec2::ZERO,
        button: MouseButton::Left,
        pressed: true,
    });
    assert_eq!(btn.state, ButtonState::Pressed);

    // Release mouse
    let clicked = btn.handle_event(&UiEvent::MouseButton {
        position: Vec2::ZERO,
        button: MouseButton::Left,
        pressed: false,
    });
    assert!(clicked);
    assert_eq!(btn.state, ButtonState::Hover);

    // Toggle button mode
    btn.toggle_mode = true;
    assert!(!btn.is_pressed());
    btn.handle_event(&UiEvent::MouseButton {
        position: Vec2::ZERO,
        button: MouseButton::Left,
        pressed: true,
    });
    assert!(btn.is_pressed());
    btn.handle_event(&UiEvent::MouseButton {
        position: Vec2::ZERO,
        button: MouseButton::Left,
        pressed: true,
    });
    assert!(!btn.is_pressed());

    // Disabled state
    btn.set_disabled(true);
    assert_eq!(btn.state, ButtonState::Disabled);
    let clicked_disabled = btn.handle_event(&UiEvent::MouseButton {
        position: Vec2::ZERO,
        button: MouseButton::Left,
        pressed: false,
    });
    assert!(!clicked_disabled);
}

#[test]
fn test_line_edit_caret_editing_and_masking() {
    let mut line_edit = LineEdit::new();
    assert_eq!(line_edit.text(), "");
    assert_eq!(line_edit.caret_column(), 0);

    // Insert text
    line_edit.insert_text("Rust");
    assert_eq!(line_edit.text(), "Rust");
    assert_eq!(line_edit.caret_column(), 4);

    line_edit.insert_text(" Engine");
    assert_eq!(line_edit.text(), "Rust Engine");
    assert_eq!(line_edit.caret_column(), 11);

    // Caret movement and backspace
    line_edit.set_caret_column(4);
    line_edit.delete_char_before_caret(); // Deletes 't'
    assert_eq!(line_edit.text(), "Rus Engine");
    assert_eq!(line_edit.caret_column(), 3);

    // Delete character at caret
    line_edit.delete_char_at_caret(); // Deletes ' '
    assert_eq!(line_edit.text(), "RusEngine");

    // Selection
    line_edit.select_all();
    assert_eq!(line_edit.selected_text(), Some("RusEngine"));

    // Secret masking
    line_edit.secret = true;
    assert_eq!(line_edit.display_text(), "*********");
}

#[test]
fn test_text_edit_multiline_editing() {
    let mut text_edit = TextEdit::new();
    text_edit.set_text("Hello\nWorld");
    assert_eq!(text_edit.line_count(), 2);
    assert_eq!(text_edit.get_line(0), Some("Hello"));
    assert_eq!(text_edit.get_line(1), Some("World"));

    // Caret line and column
    text_edit.set_caret(1, 5); // At end of "World"
    assert_eq!(text_edit.caret_line(), 1);
    assert_eq!(text_edit.caret_column(), 5);

    text_edit.insert_text("!");
    assert_eq!(text_edit.get_line(1), Some("World!"));

    // Insert newline
    text_edit.insert_text("\nNext Line");
    assert_eq!(text_edit.line_count(), 3);
    assert_eq!(text_edit.get_line(2), Some("Next Line"));
}

#[test]
fn test_range_slider_and_progress_bar() {
    let mut range = Range::new(0.0, 100.0, 5.0);
    range.set_value(50.0);
    assert_eq!(range.value(), 50.0);
    assert_eq!(range.ratio(), 0.5);

    // Step snapping
    range.set_value(53.2);
    assert_eq!(range.value(), 55.0);

    // Ratio mapping
    range.set_ratio(0.8);
    assert_eq!(range.value(), 80.0);

    // Horizontal Slider
    let mut hslider = Slider::new(SliderOrientation::Horizontal, 0.0, 10.0, 1.0);
    hslider.set_value_from_offset(30.0, 100.0); // 30px out of 100px -> ratio 0.3 -> value 3.0
    assert_eq!(hslider.range.value(), 3.0);

    // Progress Bar
    let mut pbar = ProgressBar::new(0.0, 200.0);
    pbar.set_value(150.0);
    assert_eq!(pbar.ratio(), 0.75);
    assert_eq!(pbar.percentage_string(), "75%");
    pbar.fill_mode = ProgressFillMode::RightToLeft;
    assert_eq!(pbar.fill_mode, ProgressFillMode::RightToLeft);
}

#[test]
fn test_checkbox_and_check_button() {
    let mut cb = CheckBox::new("Enable Shadows");
    assert!(!cb.is_checked());
    cb.toggle();
    assert!(cb.is_checked());
    cb.set_checked(false);
    assert!(!cb.is_checked());

    let mut switch = CheckButton::new("Fullscreen Mode");
    assert!(!switch.is_checked());
    switch.toggle();
    assert!(switch.is_checked());
}

#[test]
fn test_tree_hierarchy_and_selection() {
    let mut tree = Tree::new();
    let root = tree.create_item(None, "Root");
    let child1 = tree.create_item(Some(root), "Scene 1");
    let child2 = tree.create_item(Some(root), "Scene 2");
    let subchild = tree.create_item(Some(child1), "Player");

    assert_eq!(tree.item_count(), 4);
    assert_eq!(tree.get_item_text(subchild), Some("Player"));

    // Collapse and expand
    assert!(!tree.is_collapsed(child1));
    tree.set_collapsed(child1, true);
    assert!(tree.is_collapsed(child1));

    // Selection
    assert_eq!(tree.selected_item(), None);
    tree.select_item(child2);
    assert_eq!(tree.selected_item(), Some(child2));
}

#[test]
fn test_item_list_and_selection() {
    let mut item_list = ItemList::new();
    item_list.add_item("Asset A");
    item_list.add_item("Asset B");
    item_list.add_item("Asset C");

    assert_eq!(item_list.item_count(), 3);
    assert_eq!(item_list.get_item_text(1), Some("Asset B"));

    // Single select mode
    item_list.select_mode = ItemListSelectMode::Single;
    item_list.select(1);
    assert_eq!(item_list.selected_indices(), vec![1]);

    // Multi select mode
    item_list.select_mode = ItemListSelectMode::Multi;
    item_list.select(2);
    assert_eq!(item_list.selected_indices(), vec![1, 2]);

    item_list.deselect(1);
    assert_eq!(item_list.selected_indices(), vec![2]);
}

#[test]
fn test_popup_menu_menu_bar_and_option_button() {
    let mut menu = PopupMenu::new();
    menu.add_item(1, "New File");
    menu.add_item(2, "Open File");
    menu.add_separator();
    menu.add_check_item(3, "Show Grid", true);

    assert_eq!(menu.item_count(), 4);
    assert!(menu.get_item(2).unwrap().separator);
    assert!(menu.get_item(3).unwrap().checked);

    // Menu Bar
    let mut menu_bar = MenuBar::new();
    menu_bar.add_menu("File", menu);
    assert_eq!(menu_bar.menu_count(), 1);
    assert_eq!(menu_bar.get_menu_title(0), Some("File"));

    // Option Button (Dropdown)
    let mut option_btn = OptionButton::new();
    option_btn.add_item(0, "Low");
    option_btn.add_item(1, "Medium");
    option_btn.add_item(2, "High");
    option_btn.add_item(3, "Ultra");

    assert_eq!(option_btn.selected_index(), Some(0));
    assert_eq!(option_btn.selected_text(), Some("Low"));

    option_btn.select(2);
    assert_eq!(option_btn.selected_index(), Some(2));
    assert_eq!(option_btn.selected_text(), Some("High"));
}

#[test]
fn test_color_picker_rgba_hsv_and_hex() {
    let mut picker = ColorPicker::new();
    // Default color is white [1.0, 1.0, 1.0, 1.0]
    assert_eq!(picker.color(), [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(picker.to_hex_string(), "#FFFFFFFF");

    // Set RGBA Red
    picker.set_rgba([1.0, 0.0, 0.0, 1.0]);
    assert_eq!(picker.to_hex_string(), "#FF0000FF");
    let (h, s, v, a) = picker.to_hsva();
    assert!((h - 0.0).abs() < 1e-3);
    assert!((s - 1.0).abs() < 1e-3);
    assert!((v - 1.0).abs() < 1e-3);
    assert!((a - 1.0).abs() < 1e-3);

    // Set from Hex string
    let ok = picker.set_from_hex("#00FF00FF");
    assert!(ok);
    assert_eq!(picker.color(), [0.0, 1.0, 0.0, 1.0]);

    // Set from HSVA (Blue: H=240, S=1.0, V=1.0)
    picker.set_hsva(240.0, 1.0, 1.0, 1.0);
    let color = picker.color();
    assert!((color[0] - 0.0).abs() < 1e-3);
    assert!((color[1] - 0.0).abs() < 1e-3);
    assert!((color[2] - 1.0).abs() < 1e-3);

    // Palettes
    picker.add_preset([1.0, 1.0, 0.0, 1.0]);
    assert_eq!(picker.presets().len(), 1);
}

#[test]
fn test_widgets_as_ecs_components() {
    let mut world = World::new();
    let mut dispatcher = UiInputDispatcher::default();

    let mut ctl = Control::default();
    ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let e = world.spawn((
        ctl,
        Label::new("Title"),
        Button::new("Play"),
        LineEdit::new(),
        UiReceiver::default(),
    ));

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    assert_eq!(world.get::<Label>(e).unwrap().text(), "Title");
    assert_eq!(world.get::<Button>(e).unwrap().text(), "Play");

    // Dispatch input to LineEdit via dispatcher
    dispatcher.focus_manager.set_focus(&mut world, Some(e));
    dispatcher.inject_char(&mut world, 'X');
    assert_eq!(world.get::<UiReceiver>(e).unwrap().events.len(), 2); // FocusGained, CharInput
}
