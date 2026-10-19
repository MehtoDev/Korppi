//! AssetServer coordinating loading, asynchronous dispatch, caching, and hot reloading.

use crate::error::AssetError;
use crate::handle::{Asset, AssetHandle, AssetId, AssetLoadState};
use crate::loader::AssetLoader;
use crate::vfs::Vfs;
use korppi_core::slot_map::{Handle, SlotMap};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

struct StorageEntry<T> {
    data: Option<Arc<T>>,
    state: AssetLoadState,
}

struct TypedStorage<T: Asset> {
    slots: SlotMap<StorageEntry<T>>,
    path_to_handle: HashMap<String, Handle<StorageEntry<T>>>,
}

impl<T: Asset> Default for TypedStorage<T> {
    fn default() -> Self {
        Self {
            slots: SlotMap::new(),
            path_to_handle: HashMap::new(),
        }
    }
}

/// Central asset server coordinating loading, generational caching, and VFS resolution.
pub struct AssetServer {
    vfs: Arc<Vfs>,
    storages: Arc<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
    loaders: Arc<RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
}

impl AssetServer {
    /// Creates a new `AssetServer` using the provided VFS.
    #[must_use]
    pub fn new(vfs: Vfs) -> Self {
        Self {
            vfs: Arc::new(vfs),
            storages: Arc::new(RwLock::new(HashMap::new())),
            loaders: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Registers a custom asset loader.
    pub fn register_loader<L: AssetLoader>(&self, loader: L) {
        let loader_arc = Arc::new(loader);
        let mut loaders = self.loaders.write().unwrap();
        loaders.insert(TypeId::of::<L>(), Box::new(loader_arc));
    }

    fn get_loader<L: AssetLoader>(&self) -> Result<Arc<L>, AssetError> {
        let loaders = self.loaders.read().unwrap();
        loaders
            .get(&TypeId::of::<L>())
            .and_then(|any| any.downcast_ref::<Arc<L>>())
            .cloned()
            .ok_or(AssetError::MissingLoader(L::Asset::TYPE_NAME))
    }

    fn validate_extension<L: AssetLoader>(loader: &L, clean_path: &str) -> Result<(), AssetError> {
        let allowed = loader.extensions();
        if allowed.is_empty() {
            return Ok(());
        }
        let ext = clean_path.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
        if allowed.iter().any(|a| a.eq_ignore_ascii_case(ext)) {
            Ok(())
        } else {
            Err(AssetError::UnsupportedFormat(ext.to_ascii_lowercase()))
        }
    }

    fn get_or_create_storage<T: Asset>(&self) -> Arc<RwLock<TypedStorage<T>>> {
        let type_id = TypeId::of::<T>();
        let mut storages = self.storages.write().unwrap();

        let storage_any = storages
            .entry(type_id)
            .or_insert_with(|| Box::new(Arc::new(RwLock::new(TypedStorage::<T>::default()))));

        storage_any
            .downcast_ref::<Arc<RwLock<TypedStorage<T>>>>()
            .unwrap()
            .clone()
    }

    /// Synchronously loads and decodes an asset from the VFS.
    pub fn load_sync<L: AssetLoader>(
        &self,
        path: &str,
    ) -> Result<AssetHandle<L::Asset>, AssetError> {
        let clean_path = crate::vfs::normalize_virtual_path(path)?;
        let storage_arc = self.get_or_create_storage::<L::Asset>();

        let existing_handle = {
            let storage = storage_arc.read().unwrap();
            if let Some(&handle) = storage.path_to_handle.get(&clean_path) {
                if let Some(entry) = storage.slots.get(handle) {
                    if entry.state == AssetLoadState::Loaded && entry.data.is_some() {
                        let asset_id = AssetId::new(handle.index(), handle.generation());
                        return Ok(AssetHandle::new(asset_id, Some(clean_path)));
                    }
                }
                Some(handle)
            } else {
                None
            }
        };

        let loader = self.get_loader::<L>()?;
        Self::validate_extension(&*loader, &clean_path)?;

        let load_result = self
            .vfs
            .read(&clean_path)
            .map_err(AssetError::from)
            .and_then(|bytes| loader.load(&bytes, &clean_path));

        let mut storage = storage_arc.write().unwrap();
        match load_result {
            Ok(asset) => {
                let handle = if let Some(h) = storage
                    .path_to_handle
                    .get(&clean_path)
                    .copied()
                    .or(existing_handle)
                {
                    if let Some(entry) = storage.slots.get_mut(h) {
                        entry.data = Some(Arc::new(asset));
                        entry.state = AssetLoadState::Loaded;
                        storage.path_to_handle.insert(clean_path.clone(), h);
                        h
                    } else {
                        let new_h = storage.slots.insert(StorageEntry {
                            data: Some(Arc::new(asset)),
                            state: AssetLoadState::Loaded,
                        });
                        storage.path_to_handle.insert(clean_path.clone(), new_h);
                        new_h
                    }
                } else {
                    let new_h = storage.slots.insert(StorageEntry {
                        data: Some(Arc::new(asset)),
                        state: AssetLoadState::Loaded,
                    });
                    storage.path_to_handle.insert(clean_path.clone(), new_h);
                    new_h
                };

                let asset_id = AssetId::new(handle.index(), handle.generation());
                Ok(AssetHandle::new(asset_id, Some(clean_path)))
            }
            Err(err) => {
                if let Some(h) = storage
                    .path_to_handle
                    .get(&clean_path)
                    .copied()
                    .or(existing_handle)
                {
                    if let Some(entry) = storage.slots.get_mut(h) {
                        if entry.data.is_none() {
                            entry.state = AssetLoadState::Failed(err.to_string());
                        }
                    }
                }
                Err(err)
            }
        }
    }

    /// Asynchronously requests an asset, returning a handle immediately with state `Loading`.
    pub fn load_async<L: AssetLoader>(&self, path: &str) -> AssetHandle<L::Asset> {
        let storage_arc = self.get_or_create_storage::<L::Asset>();
        let clean_path = match crate::vfs::normalize_virtual_path(path) {
            Ok(p) => p,
            Err(e) => {
                let mut storage = storage_arc.write().unwrap();
                let handle = storage.slots.insert(StorageEntry {
                    data: None,
                    state: AssetLoadState::Failed(e.to_string()),
                });
                let asset_id = AssetId::new(handle.index(), handle.generation());
                return AssetHandle::new(asset_id, Some(path.to_string()));
            }
        };

        let handle = {
            let mut storage = storage_arc.write().unwrap();
            if let Some(&existing) = storage.path_to_handle.get(&clean_path) {
                if let Some(entry) = storage.slots.get_mut(existing) {
                    if matches!(
                        entry.state,
                        AssetLoadState::Loaded | AssetLoadState::Loading
                    ) {
                        let asset_id = AssetId::new(existing.index(), existing.generation());
                        return AssetHandle::new(asset_id, Some(clean_path));
                    }
                    entry.state = AssetLoadState::Loading;
                    existing
                } else {
                    let h = storage.slots.insert(StorageEntry {
                        data: None,
                        state: AssetLoadState::Loading,
                    });
                    storage.path_to_handle.insert(clean_path.clone(), h);
                    h
                }
            } else {
                let h = storage.slots.insert(StorageEntry {
                    data: None,
                    state: AssetLoadState::Loading,
                });
                storage.path_to_handle.insert(clean_path.clone(), h);
                h
            }
        };

        let asset_id = AssetId::new(handle.index(), handle.generation());
        let vfs = self.vfs.clone();
        let path_owned = clean_path.clone();
        let storage_clone = storage_arc.clone();
        let loaders_clone = self.loaders.clone();

        std::thread::spawn(move || {
            let loader = {
                let loaders = loaders_clone.read().unwrap();
                loaders
                    .get(&TypeId::of::<L>())
                    .and_then(|any| any.downcast_ref::<Arc<L>>())
                    .cloned()
            };

            let Some(loader) = loader else {
                let mut st = storage_clone.write().unwrap();
                if let Some(entry) = st.slots.get_mut(handle) {
                    entry.state = AssetLoadState::Failed("Missing loader".into());
                }
                return;
            };

            if let Err(e) = Self::validate_extension(&*loader, &path_owned) {
                let mut st = storage_clone.write().unwrap();
                if let Some(entry) = st.slots.get_mut(handle) {
                    entry.state = AssetLoadState::Failed(e.to_string());
                }
                return;
            }

            let bytes = match vfs.read(&path_owned) {
                Ok(b) => b,
                Err(e) => {
                    let mut st = storage_clone.write().unwrap();
                    if let Some(entry) = st.slots.get_mut(handle) {
                        entry.state = AssetLoadState::Failed(e.to_string());
                    }
                    return;
                }
            };

            match loader.load(&bytes, &path_owned) {
                Ok(asset) => {
                    let mut st = storage_clone.write().unwrap();
                    if let Some(entry) = st.slots.get_mut(handle) {
                        entry.data = Some(Arc::new(asset));
                        entry.state = AssetLoadState::Loaded;
                    }
                }
                Err(e) => {
                    let mut st = storage_clone.write().unwrap();
                    if let Some(entry) = st.slots.get_mut(handle) {
                        entry.state = AssetLoadState::Failed(e.to_string());
                    }
                }
            }
        });

        AssetHandle::new(asset_id, Some(clean_path))
    }

    /// Retrieves an immutable reference counted pointer to an asset if loaded.
    #[must_use]
    pub fn get<T: Asset>(&self, handle: &AssetHandle<T>) -> Option<Arc<T>> {
        let storage_arc = self.get_or_create_storage::<T>();
        let storage = storage_arc.read().unwrap();
        let core_handle = Handle::new(handle.id().index(), handle.id().generation());
        storage
            .slots
            .get(core_handle)
            .and_then(|entry| entry.data.clone())
    }

    /// Returns the current load state of an asset handle.
    #[must_use]
    pub fn load_state<T: Asset>(&self, handle: &AssetHandle<T>) -> AssetLoadState {
        let storage_arc = self.get_or_create_storage::<T>();
        let storage = storage_arc.read().unwrap();
        let core_handle = Handle::new(handle.id().index(), handle.id().generation());
        storage
            .slots
            .get(core_handle)
            .map_or(AssetLoadState::NotLoaded, |entry| entry.state.clone())
    }

    /// Evicts a loaded or failed asset from the server cache, invalidating its generational handle.
    pub fn unload<T: Asset>(&self, handle: &AssetHandle<T>) -> bool {
        let storage_arc = self.get_or_create_storage::<T>();
        let mut storage = storage_arc.write().unwrap();
        let core_handle = Handle::new(handle.id().index(), handle.id().generation());
        let removed = storage.slots.remove(core_handle).is_some();
        if removed {
            storage.path_to_handle.retain(|_, &mut h| h != core_handle);
        }
        removed
    }

    /// In-place reloads an asset from the VFS, updating the underlying cached data for all handles.
    pub fn reload<L: AssetLoader>(&self, path: &str) -> Result<(), AssetError> {
        let clean_path = crate::vfs::normalize_virtual_path(path)?;
        let loader = self.get_loader::<L>()?;
        Self::validate_extension(&*loader, &clean_path)?;
        let bytes = self.vfs.read(&clean_path)?;
        let asset = loader.load(&bytes, &clean_path)?;

        let storage_arc = self.get_or_create_storage::<L::Asset>();
        let mut storage = storage_arc.write().unwrap();

        if let Some(&handle) = storage.path_to_handle.get(&clean_path) {
            if let Some(entry) = storage.slots.get_mut(handle) {
                entry.data = Some(Arc::new(asset));
                entry.state = AssetLoadState::Loaded;
                return Ok(());
            }
        }

        Err(AssetError::NotFound(clean_path))
    }
}
