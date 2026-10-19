//! Regression tests for `FIX-13.5` (`SCN-01`–`SCN-04` from the `FIX-12.5` scene audit).

use korppi_ecs::{Changed, Entity, IsA, World};
use korppi_math::{Quat, Vec3};
use korppi_scene::{
    attach_child, propagate_transforms, GlobalTransform, LocalTransform, Name, Prefab,
    PrefabOverride, Scene, SceneError, SceneRegistry,
};
use std::num::NonZeroU32;

/// SCN-01 (H1): `propagate_transforms` traverses through root and intermediate entities
/// that do not carry `LocalTransform`, and only marks `GlobalTransform` as `Changed` when
/// its computed matrix actually changes.
#[test]
fn scn_01_propagate_transforms_missing_local_and_change_detection() {
    let mut world = World::new();

    // 1. Root entity with Name + GlobalTransform::default(), no LocalTransform
    let folder_root = world.spawn((Name::new("FolderRoot"), GlobalTransform::default()));
    let child = world.spawn((
        Name::new("ChildUnderFolder"),
        LocalTransform::from_translation(Vec3::new(10.0, 20.0, 30.0)),
    ));
    attach_child(&mut world, folder_root, child).unwrap();

    // 2. Root with LocalTransform -> intermediate folder with NO components -> grandchild with LocalTransform
    let root2 = world.spawn((LocalTransform::from_translation(Vec3::new(1.0, 2.0, 3.0)),));
    let mid_folder = world.spawn(());
    let grandchild = world.spawn((LocalTransform::from_translation(Vec3::new(4.0, 5.0, 6.0)),));
    attach_child(&mut world, root2, mid_folder).unwrap();
    attach_child(&mut world, mid_folder, grandchild).unwrap();

    // Frame 1
    let mut schedule = korppi_ecs::Schedule::new();
    let changed_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count_clone = changed_count.clone();
    schedule.add(
        korppi_ecs::Stage::PostUpdate,
        propagate_transforms as fn(&mut World),
    );
    schedule.add(
        korppi_ecs::Stage::Render,
        move |mut q: korppi_ecs::Query<(&GlobalTransform, Changed<GlobalTransform>)>| {
            let n = q.iter_mut().count();
            count_clone.store(n, std::sync::atomic::Ordering::SeqCst);
        },
    );

    schedule.run(&mut world);

    let child_global = world
        .get::<GlobalTransform>(child)
        .expect("child under LocalTransform-less root must receive GlobalTransform");
    assert!(
        (child_global.translation() - Vec3::new(10.0, 20.0, 30.0)).length() < 1e-5,
        "unexpected child translation: {:?}",
        child_global.translation()
    );

    let gc_global = world.get::<GlobalTransform>(grandchild).expect(
        "grandchild under LocalTransform-less intermediate node must receive GlobalTransform",
    );
    assert!(
        (gc_global.translation() - Vec3::new(5.0, 7.0, 9.0)).length() < 1e-5,
        "unexpected grandchild translation: {:?}",
        gc_global.translation()
    );

    // Frame 2: nothing moved — Changed<GlobalTransform> must NOT fire on stationary entities
    schedule.run(&mut world);
    assert_eq!(
        changed_count.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "propagate_transforms must not mark stationary GlobalTransforms as Changed"
    );
}

/// SCN-02 (H2): `GlobalTransform::scale()` and `GlobalTransform::rotation()` handle negative
/// (odd-axis reflection / mirrored) scales by preserving the negative sign and extracting a
/// valid unit quaternion.
#[test]
fn scn_02_negative_scale_decomposition() {
    let mut local = LocalTransform::from_scale(Vec3::new(-1.0, 1.0, 1.0));
    local.translation = Vec3::new(1.0, 2.0, 3.0);
    let global = GlobalTransform::from_matrix(local.compute_matrix());

    let scale = global.scale();
    let rot = global.rotation();

    assert!(
        (scale - Vec3::new(-1.0, 1.0, 1.0)).length() < 1e-5,
        "expected scale (-1, 1, 1), got {scale:?}"
    );
    assert!(
        (rot.length() - 1.0).abs() < 1e-5,
        "expected unit quaternion from reflected matrix, got {rot:?} (len={})",
        rot.length()
    );
    assert!(
        rot.dot(Quat::IDENTITY).abs() > 0.9999,
        "expected identity rotation for pure X reflection, got {rot:?}"
    );
}

/// SCN-03 (H3): `Scene::from_world` rejects duplicate entities, and `Scene::spawn`
/// validates unregistered components, duplicate entity IDs, and hierarchy cycles/self-parents
/// atomically without leaving partial entities in `World`.
#[test]
fn scn_03_scene_atomic_spawn_and_validation() {
    let mut src_world = World::new();
    let e0 = src_world.spawn((Name::new("E0"), LocalTransform::default()));
    let registry = SceneRegistry::default();

    // 1. Duplicate entities in Scene::from_world must fail
    let dup_res = Scene::from_world(&src_world, &[e0, e0], &registry);
    assert!(
        dup_res.is_err(),
        "Scene::from_world must reject duplicate entity slice, got {dup_res:?}"
    );

    // 2. Scene with an unregistered component on second entity must roll back first entity on error
    let mut scene = Scene::from_world(&src_world, &[e0], &registry).unwrap();
    let mut bad_entity = scene.entities[0].clone();
    bad_entity.entity = Entity::new(99, NonZeroU32::new(1).unwrap());
    bad_entity
        .components
        .insert("UnknownComponent".to_string(), serde_json::json!({}));
    scene.entities.push(bad_entity);

    let mut dst_world = World::new();
    let res = scene.spawn(&mut dst_world, &registry);
    assert!(matches!(res, Err(SceneError::UnregisteredComponent(_))));
    assert_eq!(
        dst_world.entity_count(),
        0,
        "Scene::spawn must not leave partial entities in World when spawning fails"
    );

    // 3. Scene JSON with self-parent cycle must be rejected cleanly without leaving partial state
    let cyclic_json = r#"{
        "entities": [
            {
                "entity": {"index": 0, "generation": 1},
                "components": {"Name": "SelfParent"},
                "children": [{"index": 0, "generation": 1}]
            }
        ]
    }"#;
    let cyclic_scene = Scene::from_json(cyclic_json).unwrap();
    let mut cycle_world = World::new();
    let cycle_res = cyclic_scene.spawn(&mut cycle_world, &registry);
    assert!(
        cycle_res.is_err(),
        "Scene::spawn must return Err on self-parent cycle"
    );
    assert_eq!(
        cycle_world.entity_count(),
        0,
        "Scene::spawn must clean up entities on hierarchy failure"
    );
}

/// SCN-04 (H4): `Prefab::instantiate_with_overrides` returns a dedicated error (not `UnregisteredComponent`)
/// on out-of-bounds `entity_index`, and `Scene` round-trips `IsA` relationships without
/// flattening inherited base components onto instances.
#[test]
fn scn_04_prefab_bounds_and_isa_roundtrip() {
    let mut world = World::new();
    let base = world.spawn((
        Name::new("BaseTemplate"),
        LocalTransform::from_translation(Vec3::new(1.0, 2.0, 3.0)),
    ));
    let instance = world.spawn((Name::new("DerivedInstance"),));
    world.add_pair::<IsA>(instance, base);

    let registry = SceneRegistry::default();
    let scene = Scene::from_world(&world, &[base, instance], &registry).unwrap();

    // Verify inherited LocalTransform was not flattened onto DerivedInstance in Scene
    assert!(
        !scene.entities[1].components.contains_key("LocalTransform"),
        "Scene::from_world must not flatten inherited IsA components onto derived entity"
    );

    // 1. Out-of-bounds PrefabOverride::entity_index must NOT be misreported as UnregisteredComponent
    let prefab = Prefab::new("TestPrefab", scene.clone());
    let mut target_world = World::new();
    let bad_override = PrefabOverride::new(999)
        .with_component(
            "LocalTransform",
            &LocalTransform::from_translation(Vec3::new(9.0, 9.0, 9.0)),
        )
        .unwrap();
    let oob_res = prefab.instantiate_with_overrides(&mut target_world, &registry, &[bad_override]);
    assert!(oob_res.is_err());
    assert!(
        !matches!(oob_res, Err(SceneError::UnregisteredComponent(_))),
        "out-of-bounds entity_index must not return UnregisteredComponent: {oob_res:?}"
    );

    // 2. IsA relationship must survive Scene JSON round-trip
    let json = scene.to_json().unwrap();
    let restored = Scene::from_json(&json).unwrap();
    let mut dst_world = World::new();
    let spawned = restored.spawn(&mut dst_world, &registry).unwrap();
    let new_base = spawned[0];
    let new_inst = spawned[1];

    assert!(
        dst_world.has_pair::<IsA>(new_inst, new_base),
        "IsA relationship must be preserved across Scene serialization round-trip"
    );
    assert_eq!(dst_world.targets_of::<IsA>(new_inst), &[new_base]);
    assert_eq!(
        dst_world
            .get::<LocalTransform>(new_inst)
            .unwrap()
            .translation,
        Vec3::new(1.0, 2.0, 3.0)
    );
}
