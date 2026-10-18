//! Hierarchical tree view widget (`Tree`) for outliners and scene graphs.

use korppi_ecs::Component;

/// Individual node entry within a `Tree`.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeItem {
    /// Unique item index.
    pub id: usize,
    /// Item label text.
    pub text: String,
    /// Optional item icon path or identifier.
    pub icon: Option<String>,
    /// Parent item index, or None if root.
    pub parent: Option<usize>,
    /// Indices of child tree items.
    pub children: Vec<usize>,
    /// Whether item children are collapsed and hidden.
    pub collapsed: bool,
    /// Whether this item is currently selected.
    pub selected: bool,
}

/// Hierarchical multi-level tree control widget component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tree {
    /// Arena storing all tree items by unique ID index.
    items: Vec<TreeItem>,
    /// List of root item indices.
    roots: Vec<usize>,
    /// Currently selected tree item ID.
    selected_item: Option<usize>,
}

impl Component for Tree {}

impl Tree {
    /// Creates a new empty `Tree`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new tree item with an optional parent.
    ///
    /// If `parent` is `None`, adds the item as a root.
    pub fn create_item(&mut self, parent: Option<usize>, text: impl Into<String>) -> usize {
        let id = self.items.len();
        let item = TreeItem {
            id,
            text: text.into(),
            icon: None,
            parent,
            children: Vec::new(),
            collapsed: false,
            selected: false,
        };
        self.items.push(item);

        if let Some(p_id) = parent {
            if let Some(p) = self.items.get_mut(p_id) {
                p.children.push(id);
            }
        } else {
            self.roots.push(id);
        }

        id
    }

    /// Returns the total number of items in the tree.
    #[inline]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Returns text of the specified item.
    pub fn get_item_text(&self, id: usize) -> Option<&str> {
        self.items.get(id).map(|item| item.text.as_str())
    }

    /// Returns whether the specified item is collapsed.
    pub fn is_collapsed(&self, id: usize) -> bool {
        self.items
            .get(id)
            .map(|item| item.collapsed)
            .unwrap_or(false)
    }

    /// Sets the collapsed state of the specified item.
    pub fn set_collapsed(&mut self, id: usize, collapsed: bool) {
        if let Some(item) = self.items.get_mut(id) {
            item.collapsed = collapsed;
        }
    }

    /// Selects the specified item, deselecting previous selection.
    pub fn select_item(&mut self, id: usize) {
        if let Some(prev) = self.selected_item {
            if let Some(item) = self.items.get_mut(prev) {
                item.selected = false;
            }
        }

        if let Some(item) = self.items.get_mut(id) {
            item.selected = true;
            self.selected_item = Some(id);
        }
    }

    /// Returns the currently selected item ID.
    #[inline]
    pub fn selected_item(&self) -> Option<usize> {
        self.selected_item
    }

    /// Returns slice of root item IDs.
    #[inline]
    pub fn root_items(&self) -> &[usize] {
        &self.roots
    }

    /// Returns slice of child IDs for the specified item.
    pub fn children_of(&self, id: usize) -> &[usize] {
        self.items
            .get(id)
            .map(|item| item.children.as_slice())
            .unwrap_or(&[])
    }
}
