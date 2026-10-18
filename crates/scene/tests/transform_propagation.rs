//! Integration tests for local-to-world transform propagation in scene hierarchies.

use korppi_ecs::{Schedule, Stage, World};
use korppi_math::Vec3;
use korppi_scene::hierarchy::attach_child;
use korppi_scene::transform::{propagate_transforms, GlobalTransform, LocalTransform};

#[test]
fn root_entity_global_transform_matches_local() {
    let mut world = World::new();

    let local = LocalTransform::from_translation(Vec3::new(5.0, -3.0, 2.0));
    let entity = world.spawn((local, GlobalTransform::default()));

    propagate_transforms(&mut world);

    let global = world.get::<GlobalTransform>(entity).unwrap();
    assert_eq!(global.translation(), Vec3::new(5.0, -3.0, 2.0));
}

#[test]
fn child_inherits_parent_translation_rotation_scale() {
    let mut world = World::new();

    // Parent translated by (10, 0, 0)
    let parent_local = LocalTransform::from_translation(Vec3::new(10.0, 0.0, 0.0));
    let parent = world.spawn((parent_local, GlobalTransform::default()));

    // Child translated locally by (0, 5, 0)
    let child_local = LocalTransform::from_translation(Vec3::new(0.0, 5.0, 0.0));
    let child = world.spawn((child_local, GlobalTransform::default()));

    attach_child(&mut world, parent, child).unwrap();

    propagate_transforms(&mut world);

    let parent_global = world.get::<GlobalTransform>(parent).unwrap();
    assert_eq!(parent_global.translation(), Vec3::new(10.0, 0.0, 0.0));

    let child_global = world.get::<GlobalTransform>(child).unwrap();
    assert_eq!(child_global.translation(), Vec3::new(10.0, 5.0, 0.0));
}

#[test]
fn deep_hierarchy_propagates_compounded_transforms() {
    let mut world = World::new();

    // Root at (10, 0, 0)
    let root = world.spawn((
        LocalTransform::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        GlobalTransform::default(),
    ));

    // Child scaled by 2.0
    let child = world.spawn((
        LocalTransform::from_scale(Vec3::splat(2.0)),
        GlobalTransform::default(),
    ));

    // Grandchild translated locally by (1, 0, 0)
    let grandchild = world.spawn((
        LocalTransform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        GlobalTransform::default(),
    ));

    attach_child(&mut world, root, child).unwrap();
    attach_child(&mut world, child, grandchild).unwrap();

    propagate_transforms(&mut world);

    let grandchild_global = world.get::<GlobalTransform>(grandchild).unwrap();
    // Translation: 10 + (1 * 2) = 12
    assert!((grandchild_global.translation() - Vec3::new(12.0, 0.0, 0.0)).length() < 1e-4);
    assert_eq!(grandchild_global.scale(), Vec3::splat(2.0));
}

#[test]
fn ecs_schedule_post_update_runs_propagation() {
    let mut world = World::new();
    let mut schedule = Schedule::new();

    let parent = world.spawn((
        LocalTransform::from_translation(Vec3::ZERO),
        GlobalTransform::default(),
    ));

    let child = world.spawn((
        LocalTransform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        GlobalTransform::default(),
    ));

    attach_child(&mut world, parent, child).unwrap();

    // Register propagation system in PostUpdate
    schedule.add_system(Stage::PostUpdate, propagate_transforms);

    // Initial run
    schedule.run_stage(Stage::PostUpdate, &mut world);
    assert_eq!(
        world.get::<GlobalTransform>(child).unwrap().translation(),
        Vec3::new(1.0, 0.0, 0.0)
    );

    // Mutate parent local transform
    if let Some(parent_local) = world.get_mut::<LocalTransform>(parent) {
        parent_local.translation = Vec3::new(50.0, 0.0, 0.0);
    }

    // Run PostUpdate stage again
    schedule.run_stage(Stage::PostUpdate, &mut world);

    // Child global transform should now be 50 + 1 = 51
    assert_eq!(
        world.get::<GlobalTransform>(child).unwrap().translation(),
        Vec3::new(51.0, 0.0, 0.0)
    );
}
