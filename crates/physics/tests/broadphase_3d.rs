//! Primary integration test for 3D broadphase three-tree architecture and pair generation (P3D-19).

use korppi_math::Vec3;
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::shape::Shape;
use korppi_physics::PhysicsWorld;

#[test]
fn test_three_tree_broadphase_eliminates_kinematic_static_pairs() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    // Static body
    let s = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 0.0, 0.0)));
    world.create_collider(
        s,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        }),
    );

    // Kinematic body overlapping the static body
    let k = world.create_body(RigidBody::new(
        BodyType::Kinematic,
        Vec3::new(0.5, 0.0, 0.0),
    ));
    world.create_collider(
        k,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        }),
    );

    // Second kinematic body overlapping k
    let k2 = world.create_body(RigidBody::new(
        BodyType::Kinematic,
        Vec3::new(0.2, 0.0, 0.0),
    ));
    world.create_collider(
        k2,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        }),
    );

    world.step(1.0 / 60.0);

    // Broadphase candidate pairs must NEVER contain kinematic-static or kinematic-kinematic pairs
    let pairs = world.broadphase_candidate_pairs();
    for &(c_a, c_b) in pairs {
        let body_a = world.collider_body(c_a).unwrap();
        let body_b = world.collider_body(c_b).unwrap();
        let type_a = world.body(body_a).unwrap().body_type;
        let type_b = world.body(body_b).unwrap().body_type;

        assert!(
            type_a == BodyType::Dynamic || type_b == BodyType::Dynamic,
            "Broadphase pairs must involve at least one dynamic body! Found {type_a:?} vs {type_b:?}"
        );
    }
}

#[test]
fn test_three_tree_broadphase_dynamic_pairs_generated_correctly() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    // Static floor
    let s = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    let col_s = world.create_collider(
        s,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(5.0),
        }),
    );

    // Moving dynamic body overlapping static floor
    let d = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.0, 0.0))
            .with_linear_velocity(Vec3::new(1.0, 0.0, 0.0)),
    );
    let col_d = world.create_collider(
        d,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        }),
    );

    world.step(1.0 / 60.0);

    let pairs = world.broadphase_candidate_pairs();
    let expected_pair = (
        col_s.index().min(col_d.index()),
        col_s.index().max(col_d.index()),
    );
    assert!(
        pairs.contains(&expected_pair),
        "Dynamic body must generate broadphase pair with static body"
    );
}

#[test]
fn test_broadphase_moved_proxy_tracking() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    // 10 resting dynamic bodies separated far apart
    let mut bodies = Vec::new();
    for i in 0..10 {
        let b = world.create_body(RigidBody::new(
            BodyType::Dynamic,
            Vec3::new((i * 10) as f32, 0.0, 0.0),
        ));
        world.create_collider(b, Collider::new(Shape::Sphere { radius: 1.0 }));
        bodies.push(b);
    }

    world.step(1.0 / 60.0);

    // After step 1, proxies are stationary. Move only body 0 towards body 1.
    world.body_mut(bodies[0]).unwrap().linear_velocity = Vec3::new(100.0, 0.0, 0.0);

    world.step(1.0 / 60.0);

    // Broadphase tree proxy heights and node count queries
    assert!(world.broadphase_tree_height(BodyType::Dynamic) > 0);
    assert!(world.broadphase_tree_height(BodyType::Static) == 0);
}
