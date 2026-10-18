//! Item list widget (`ItemList`) for asset browsers and item selection grids.

use korppi_ecs::Component;

/// Item selection behavior mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ItemListSelectMode {
    /// Only a single item may be selected at a time.
    #[default]
    Single,
    /// Multiple items can be selected concurrently.
    Multi,
}

/// Entry item stored in an `ItemList`.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemListItem {
    /// Item label text.
    pub text: String,
    /// Optional item icon path.
    pub icon: Option<String>,
    /// Whether this item can be selected.
    pub selectable: bool,
    /// Whether this item is currently selected.
    pub selected: bool,
    /// Whether user interaction is disabled.
    pub disabled: bool,
}

/// Item list widget component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ItemList {
    /// List of stored items.
    items: Vec<ItemListItem>,
    /// Selection interaction mode.
    pub select_mode: ItemListSelectMode,
}

impl Component for ItemList {}

impl ItemList {
    /// Creates a new empty `ItemList`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new item to the list and returns its index.
    pub fn add_item(&mut self, text: impl Into<String>) -> usize {
        let index = self.items.len();
        self.items.push(ItemListItem {
            text: text.into(),
            icon: None,
            selectable: true,
            selected: false,
            disabled: false,
        });
        index
    }

    /// Returns the total number of items.
    #[inline]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Returns text of the item at `index`.
    pub fn get_item_text(&self, index: usize) -> Option<&str> {
        self.items.get(index).map(|item| item.text.as_str())
    }

    /// Selects the item at `index`.
    pub fn select(&mut self, index: usize) {
        if index >= self.items.len() || !self.items[index].selectable || self.items[index].disabled
        {
            return;
        }

        if self.select_mode == ItemListSelectMode::Single {
            for item in &mut self.items {
                item.selected = false;
            }
        }

        self.items[index].selected = true;
    }

    /// Deselects the item at `index`.
    pub fn deselect(&mut self, index: usize) {
        if let Some(item) = self.items.get_mut(index) {
            item.selected = false;
        }
    }

    /// Returns indices of all currently selected items.
    pub fn selected_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter_map(|(i, item)| if item.selected { Some(i) } else { None })
            .collect()
    }

    /// Clears all items from the list.
    pub fn clear(&mut self) {
        self.items.clear();
    }
}
