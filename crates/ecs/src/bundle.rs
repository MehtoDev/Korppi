//! Component bundle trait for spawning and inserting tuples of components.

use crate::archetype::Archetype;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::tick::Tick;

/// A collection of components that can be inserted into an entity as a single bundle.
pub trait Bundle: Sized + 'static {
    /// Registers all component types in this bundle and returns their sorted IDs.
    fn register(registry: &mut ComponentRegistry) -> Vec<ComponentId>;

    /// Writes all component values in this bundle into their corresponding archetype columns,
    /// stamped as added and changed at `tick`.
    fn write_to_archetype(
        self,
        archetype: &mut Archetype,
        registry: &ComponentRegistry,
        tick: Tick,
    );
}

impl Bundle for () {
    fn register(_registry: &mut ComponentRegistry) -> Vec<ComponentId> {
        Vec::new()
    }

    fn write_to_archetype(
        self,
        _archetype: &mut Archetype,
        _registry: &ComponentRegistry,
        _tick: Tick,
    ) {
    }
}

macro_rules! impl_bundle {
    ($($name:ident),*) => {
        impl<$($name: Component),*> Bundle for ($($name,)*) {
            #[allow(unused_mut)]
            fn register(registry: &mut ComponentRegistry) -> Vec<ComponentId> {
                let raw_ids = [$(registry.register::<$name>()),*];
                let mut ids = raw_ids.to_vec();
                ids.sort();
                for i in 1..ids.len() {
                    if ids[i - 1] == ids[i] {
                        let dup_id = ids[i];
                        let name = registry
                            .get_info(dup_id)
                            .map(|info| info.name())
                            .unwrap_or("unknown");
                        panic!("Duplicate component type `{}` in bundle", name);
                    }
                }
                ids
            }

            #[allow(non_snake_case)]
            fn write_to_archetype(
                self,
                archetype: &mut Archetype,
                registry: &ComponentRegistry,
                tick: Tick,
            ) {
                let ($($name,)*) = self;
                $(
                    let id = registry.id_of::<$name>().expect("Component not registered");
                    let column = archetype.column_mut(id).expect("Missing column for bundle component");
                    // SAFETY: $name type matches the registered component layout and type ID for this column.
                    unsafe {
                        column.push($name, tick);
                    }
                )*
            }
        }
    };
}

impl_bundle!(A);
impl_bundle!(A, B);
impl_bundle!(A, B, C);
impl_bundle!(A, B, C, D);
impl_bundle!(A, B, C, D, E);
impl_bundle!(A, B, C, D, E, F);
impl_bundle!(A, B, C, D, E, F, G);
impl_bundle!(A, B, C, D, E, F, G, H);
impl_bundle!(A, B, C, D, E, F, G, H, I);
impl_bundle!(A, B, C, D, E, F, G, H, I, J);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K);
impl_bundle!(A, B, C, D, E, F, G, H, I, J, K, L);
