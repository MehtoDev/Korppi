//! Serde adapters for the world's ids, used with `#[serde(with = ...)]`.
//!
//! `korppi_core`'s [`Handle`] and [`IdPool`] carry no serde implementations (the dependency is
//! an open decision there, see `docs/rules/2-dependencies.md`), so snapshots save a handle as
//! its `(index, generation)` and a pool through [`IdPool::to_parts`]. The contact store's pair
//! map is saved as a list sorted by pair, so a snapshot's bytes do not depend on hashing.

use korppi_core::{Handle, IdPool, IdPoolParts};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::num::NonZeroU32;

/// A handle as `(index, generation)`.
fn raw<T>(handle: Handle<T>) -> (u32, u32) {
    (handle.index(), handle.generation().get())
}

/// The handle saved as `raw`, or `None` for a zero generation.
fn cook<T>((index, generation): (u32, u32)) -> Option<Handle<T>> {
    Some(Handle::new(index, NonZeroU32::new(generation)?))
}

fn invalid<E: serde::de::Error>() -> E {
    E::custom("a handle with generation zero")
}

/// One handle.
pub mod handle {
    use super::*;

    pub fn serialize<T, S: Serializer>(
        handle: &Handle<T>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        raw(*handle).serialize(serializer)
    }

    pub fn deserialize<'de, T, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Handle<T>, D::Error> {
        cook(<(u32, u32)>::deserialize(deserializer)?).ok_or_else(invalid)
    }
}

/// A list of handles.
pub mod handle_vec {
    use super::*;

    pub fn serialize<T, S: Serializer>(
        handles: &[Handle<T>],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(handles.iter().map(|&h| raw(h)))
    }

    pub fn deserialize<'de, T, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<Handle<T>>, D::Error> {
        Vec::<(u32, u32)>::deserialize(deserializer)?
            .into_iter()
            .map(|h| cook(h).ok_or_else(invalid))
            .collect()
    }
}

/// A list of optional handles.
pub mod option_handle_vec {
    use super::*;

    pub fn serialize<T, S: Serializer>(
        handles: &[Option<Handle<T>>],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(handles.iter().map(|h| h.map(raw)))
    }

    pub fn deserialize<'de, T, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<Option<Handle<T>>>, D::Error> {
        Vec::<Option<(u32, u32)>>::deserialize(deserializer)?
            .into_iter()
            .map(|h| h.map(|h| cook(h).ok_or_else(invalid)).transpose())
            .collect()
    }
}

/// A list of handle lists.
pub mod nested_handle_vec {
    use super::*;

    pub fn serialize<T, S: Serializer>(
        lists: &[Vec<Handle<T>>],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            lists
                .iter()
                .map(|list| list.iter().map(|&h| raw(h)).collect::<Vec<_>>()),
        )
    }

    pub fn deserialize<'de, T, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<Vec<Handle<T>>>, D::Error> {
        Vec::<Vec<(u32, u32)>>::deserialize(deserializer)?
            .into_iter()
            .map(|list| {
                list.into_iter()
                    .map(|h| cook(h).ok_or_else(invalid))
                    .collect()
            })
            .collect()
    }
}

/// An id pool, through its parts.
pub mod id_pool {
    use super::*;

    #[derive(Serialize, Deserialize)]
    struct Parts {
        generations: Vec<u32>,
        live: Vec<bool>,
        free: Vec<u32>,
    }

    pub fn serialize<T, S: Serializer>(pool: &IdPool<T>, serializer: S) -> Result<S::Ok, S::Error> {
        let IdPoolParts {
            generations,
            live,
            free,
        } = pool.to_parts();
        Parts {
            generations,
            live,
            free,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, T, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<IdPool<T>, D::Error> {
        let Parts {
            generations,
            live,
            free,
        } = Parts::deserialize(deserializer)?;
        IdPool::from_parts(IdPoolParts {
            generations,
            live,
            free,
        })
        .ok_or_else(|| D::Error::custom("an inconsistent id pool"))
    }
}

/// The contact store's map from collider pair to slot, as a list sorted by pair.
pub mod pair_map {
    use super::*;

    pub fn serialize<S: Serializer>(
        map: &HashMap<(u32, u32), u32>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut entries: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_unstable();
        entries.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<HashMap<(u32, u32), u32>, D::Error> {
        Ok(Vec::<((u32, u32), u32)>::deserialize(deserializer)?
            .into_iter()
            .collect())
    }
}
