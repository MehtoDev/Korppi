//! Popup menu, menu bar, and option dropdown button widgets.

use korppi_ecs::Component;

/// Individual entry in a `PopupMenu`.
#[derive(Clone, Debug, PartialEq)]
pub struct MenuItem {
    /// Item identifier.
    pub id: u32,
    /// Item text label.
    pub text: String,
    /// Optional icon texture path.
    pub icon: Option<String>,
    /// Optional shortcut label (e.g. `"Ctrl+S"`).
    pub shortcut: Option<String>,
    /// Whether user interaction is disabled.
    pub disabled: bool,
    /// Whether this entry represents a dividing horizontal separator.
    pub separator: bool,
    /// Checked toggle state for checkable menu items.
    pub checked: bool,
    /// Whether this entry is a checkable toggle item.
    pub checkable: bool,
    /// Optional nested submenu items.
    pub submenu: Option<Vec<MenuItem>>,
}

/// Context popup menu widget component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PopupMenu {
    /// Stored menu items.
    items: Vec<MenuItem>,
    /// Whether the popup menu is currently shown on screen.
    pub visible: bool,
}

impl Component for PopupMenu {}

impl PopupMenu {
    /// Creates a new empty `PopupMenu`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a standard clickable menu item.
    pub fn add_item(&mut self, id: u32, text: impl Into<String>) {
        self.items.push(MenuItem {
            id,
            text: text.into(),
            icon: None,
            shortcut: None,
            disabled: false,
            separator: false,
            checked: false,
            checkable: false,
            submenu: None,
        });
    }

    /// Appends a dividing line separator.
    pub fn add_separator(&mut self) {
        self.items.push(MenuItem {
            id: 0,
            text: String::new(),
            icon: None,
            shortcut: None,
            disabled: true,
            separator: true,
            checked: false,
            checkable: false,
            submenu: None,
        });
    }

    /// Appends a checkable toggle item.
    pub fn add_check_item(&mut self, id: u32, text: impl Into<String>, checked: bool) {
        self.items.push(MenuItem {
            id,
            text: text.into(),
            icon: None,
            shortcut: None,
            disabled: false,
            separator: false,
            checked,
            checkable: true,
            submenu: None,
        });
    }

    /// Returns the total number of items including separators.
    #[inline]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Returns reference to the item at `index`.
    pub fn get_item(&self, index: usize) -> Option<&MenuItem> {
        self.items.get(index)
    }

    /// Clears all menu items.
    pub fn clear(&mut self) {
        self.items.clear();
    }
}

/// Top application menu bar widget component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MenuBar {
    /// Ordered list of (Title, PopupMenu) entries.
    menus: Vec<(String, PopupMenu)>,
}

impl Component for MenuBar {}

impl MenuBar {
    /// Creates a new empty `MenuBar`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a named menu dropdown.
    pub fn add_menu(&mut self, title: impl Into<String>, menu: PopupMenu) {
        self.menus.push((title.into(), menu));
    }

    /// Returns the number of menus in the menu bar.
    #[inline]
    pub fn menu_count(&self) -> usize {
        self.menus.len()
    }

    /// Returns title of menu at `index`.
    pub fn get_menu_title(&self, index: usize) -> Option<&str> {
        self.menus.get(index).map(|(t, _)| t.as_str())
    }

    /// Returns reference to `PopupMenu` at `index`.
    pub fn get_menu(&self, index: usize) -> Option<&PopupMenu> {
        self.menus.get(index).map(|(_, m)| m)
    }
}

/// Dropdown selection option button widget component (`OptionButton`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OptionButton {
    /// Selectable options `(id, label)`.
    items: Vec<(u32, String)>,
    /// Currently selected item index.
    selected_index: Option<usize>,
}

impl Component for OptionButton {}

impl OptionButton {
    /// Creates a new empty `OptionButton`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a selectable option.
    pub fn add_item(&mut self, id: u32, text: impl Into<String>) {
        let is_first = self.items.is_empty();
        self.items.push((id, text.into()));
        if is_first {
            self.selected_index = Some(0);
        }
    }

    /// Selects option by index.
    pub fn select(&mut self, index: usize) {
        if index < self.items.len() {
            self.selected_index = Some(index);
        }
    }

    /// Returns currently selected index.
    #[inline]
    pub fn selected_index(&self) -> Option<usize> {
        self.selected_index
    }

    /// Returns label of currently selected option.
    pub fn selected_text(&self) -> Option<&str> {
        let idx = self.selected_index?;
        self.items.get(idx).map(|(_, t)| t.as_str())
    }

    /// Returns ID of currently selected option.
    pub fn selected_id(&self) -> Option<u32> {
        let idx = self.selected_index?;
        self.items.get(idx).map(|(id, _)| *id)
    }

    /// Returns total option count.
    #[inline]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }
}
