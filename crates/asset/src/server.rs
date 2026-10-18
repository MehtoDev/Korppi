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

    /// Registers a custom asset loader for its target asset type.
    pub fn register_loader<L: AssetLoader>(&self, loader: L) {
        let type_id = TypeId::of::<L::Asset>();
        self.loaders
            .write()
            .unwrap()
            .insert(type_id, Box::new(loader));
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
        let storage_arc = self.get_or_create_storage::<L::Asset>();
        let mut storage = storage_arc.write().unwrap();

        if let Some(&handle) = storage.path_to_handle.get(path) {
            let asset_id = AssetId::new(handle.index(), handle.generation());
            return Ok(AssetHandle::new(asset_id, Some(path.to_string())));
        }

        let bytes = self.vfs.read(path)?;
        let loaders = self.loaders.read().unwrap();
        let loader = loaders
            .get(&TypeId::of::<L::Asset>())
            .and_then(|any| any.downcast_ref::<L>())
            .ok_or(AssetError::MissingLoader(L::Asset::TYPE_NAME))?;

        let asset = loader.load(&bytes, path)?;
        let handle = storage.slots.insert(StorageEntry {
            data: Some(Arc::new(asset)),
            state: AssetLoadState::Loaded,
        });

        storage.path_to_handle.insert(path.to_string(), handle);
        let asset_id = AssetId::new(handle.index(), handle.generation());
        Ok(AssetHandle::new(asset_id, Some(path.to_string())))
    }

    /// Asynchronously requests an asset, returning a handle immediately with state `Loading`.
    pub fn load_async<L: AssetLoader>(&self, path: &str) -> AssetHandle<L::Asset> {
        let storage_arc = self.get_or_create_storage::<L::Asset>();
        let mut storage = storage_arc.write().unwrap();

        if let Some(&handle) = storage.path_to_handle.get(path) {
            let asset_id = AssetId::new(handle.index(), handle.generation());
            return AssetHandle::new(asset_id, Some(path.to_string()));
        }

        let handle = storage.slots.insert(StorageEntry {
            data: None,
            state: AssetLoadState::Loading,
        });
        storage.path_to_handle.insert(path.to_string(), handle);

        let asset_id = AssetId::new(handle.index(), handle.generation());
        let vfs = self.vfs.clone();
        let path_owned = path.to_string();
        let storage_clone = storage_arc.clone();
        let loaders_clone = self.loaders.clone();

        std::thread::spawn(move || {
            let bytes_res = vfs.read(&path_owned);
            let bytes = match bytes_res {
                Ok(b) => b,
                Err(e) => {
                    let mut st = storage_clone.write().unwrap();
                    if let Some(entry) = st.slots.get_mut(handle) {
                        entry.state = AssetLoadState::Failed(e.to_string());
                    }
                    return;
                }
            };

            let loaders = loaders_clone.read().unwrap();
            let loader = match loaders
                .get(&TypeId::of::<L::Asset>())
                .and_then(|any| any.downcast_ref::<L>())
            {
                Some(l) => l,
                None => {
                    let mut st = storage_clone.write().unwrap();
                    if let Some(entry) = st.slots.get_mut(handle) {
                        entry.state = AssetLoadState::Failed("Missing loader".into());
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

        AssetHandle::new(asset_id, Some(path.to_string()))
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

    /// In-place reloads an asset from the VFS, updating the underlying cached data for all handles.
    pub fn reload<L: AssetLoader>(&self, path: &str) -> Result<(), AssetError> {
        let bytes = self.vfs.read(path)?;
        let loaders = self.loaders.read().unwrap();
        let loader = loaders
            .get(&TypeId::of::<L::Asset>())
            .and_then(|any| any.downcast_ref::<L>())
            .ok_or(AssetError::MissingLoader(L::Asset::TYPE_NAME))?;

        let asset = loader.load(&bytes, path)?;
        let storage_arc = self.get_or_create_storage::<L::Asset>();
        let mut storage = storage_arc.write().unwrap();

        if let Some(&handle) = storage.path_to_handle.get(path) {
            if let Some(entry) = storage.slots.get_mut(handle) {
                entry.data = Some(Arc::new(asset));
                entry.state = AssetLoadState::Loaded;
                return Ok(());
            }
        }

        Err(AssetError::NotFound(path.to_string()))
    }
}
