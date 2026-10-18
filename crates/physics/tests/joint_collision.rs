//! Bodies joined by a joint do not collide with each other unless the joint allows it (P3D-14).
//!
//! Box3D joints have `collideConnected`, false by default: the broadphase drops the pairs of
//! bodies joined by a joint that does not allow collision, and creating such a joint destroys
//! the contacts between its bodies. Before FIX-3.1 jointed bodies always collided, so ragdoll
//! limbs fought their joints.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_1_joints.c`):
//! 60 Hz, 4 substeps, 1 kg boxes, friction 0.6, no damping and no sleep.

use korppi_math::Vec3;
use korppi_physics::joint::{Joint, SphericalJoint};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};

const DT: f32 = 1.0 / 60.0;

fn add_box(world: &mut PhysicsWorld, body_type: BodyType, position: Vec3) -> u32 {
    let mut body = RigidBody::new(body_type, position).with_enable_sleep(false);
    body.linear_damping = 0.0;
    body.angular_damping = 0.0;
    let id = world.create_body(body);
    world.create_collider(
        id,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.6)
        .with_mass(1.0),
    );
    id.into()
}

fn touches(
    world: &PhysicsWorld,
    a: impl Into<korppi_physics::BodyId3d>,
    b: impl Into<korppi_physics::BodyId3d>,
) -> bool {
    let (a, b) = (a.into().index(), b.into().index());
    world
        .manifolds()
        .iter()
        .any(|m| (m.body_a, m.body_b) == (a, b) || (m.body_a, m.body_b) == (b, a))
}

/// A dynamic box half inside a static box, pinned to it at the overlap's centre, without
/// gravity. Returns the world, both boxes, and the joint.
fn pinned_overlapping_boxes(collide_connected: bool) -> (PhysicsWorld, u32, u32, u32) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);
    let fixed = add_box(&mut world, BodyType::Static, Vec3::ZERO);
    let pinned = add_box(&mut world, BodyType::Dynamic, Vec3::new(0.5, 0.0, 0.0));
    let joint = world.create_joint(
        Joint::Spherical(SphericalJoint::new(
            fixed,
            pinned,
            Vec3::new(0.25, 0.0, 0.0),
            Vec3::new(-0.25, 0.0, 0.0),
        ))
        .with_collide_connected(collide_connected),
    );
    (world, fixed, pinned, joint.into())
}

/// The largest linear plus angular speed of `body` over `steps` steps.
fn largest_speed(world: &mut PhysicsWorld, body: u32, steps: usize) -> f32 {
    let mut largest: f32 = 0.0;
    for _ in 0..steps {
        world.step(DT);
        let b = world.body(body).unwrap();
        largest = largest.max(b.linear_velocity.length() + b.angular_velocity.length());
    }
    largest
}

#[test]
fn jointed_bodies_do_not_collide() {
    let (mut world, fixed, pinned, joint) = pinned_overlapping_boxes(false);
    assert!(!world.joint(joint).unwrap().collide_connected());

    // Box3D: the pinned box never moves (largest speed 0.00000).
    let largest = largest_speed(&mut world, pinned, 60);
    assert!(
        largest <= 0.001,
        "the pinned box should stay put, the overlap pushed it to {largest} m/s"
    );
    assert!(!touches(&world, fixed, pinned));
}

#[test]
fn jointed_bodies_collide_when_the_joint_allows_it() {
    let (mut world, _, pinned, _) = pinned_overlapping_boxes(true);

    // Box3D: the overlap pushes the pinned box, which swings about its pin at up to 0.39 m/s
    // plus rad/s.
    let largest = largest_speed(&mut world, pinned, 60);
    assert!(
        largest >= 0.1,
        "the overlapping boxes should push apart, the pinned box reached only {largest}"
    );
}

#[test]
fn changing_collide_connected_takes_effect_on_the_next_step() {
    let (mut world, fixed, pinned, joint) = pinned_overlapping_boxes(true);
    world.step(DT);
    assert!(touches(&world, fixed, pinned));

    world.joint_mut(joint).unwrap().base_mut().collide_connected = false;
    world.step(DT);
    assert!(!touches(&world, fixed, pinned));
}

#[test]
fn creating_a_joint_removes_the_contact_between_its_bodies() {
    // A box resting, asleep, on the ground keeps its contact while both rest; a joint that does
    // not allow collision removes it at once, as Box3D's b3CreateJoint does.
    let mut world = PhysicsWorld::new();
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(5.0, 0.5, 5.0),
        })
        .with_mass(1.0),
    );
    let resting = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)));
    world.create_collider(
        resting,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_mass(1.0),
    );
    for _ in 0..120 {
        world.step(DT);
    }
    assert!(world.body(resting).unwrap().is_sleeping);
    assert!(touches(&world, ground, resting));

    world.create_joint(Joint::Spherical(SphericalJoint::new(
        ground,
        resting,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::ZERO,
    )));
    assert!(!touches(&world, ground, resting));
    world.step(DT);
    assert!(!touches(&world, ground, resting));
}

#[test]
fn jointed_bodies_still_collide_with_other_bodies() {
    // Two boxes joined edge to edge and dropped on the ground both land on it.
    let mut world = PhysicsWorld::new();
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(5.0, 0.5, 5.0),
        })
        .with_mass(1.0),
    );
    let left = add_box(&mut world, BodyType::Dynamic, Vec3::new(-0.5, 1.0, 0.0));
    let right = add_box(&mut world, BodyType::Dynamic, Vec3::new(0.5, 1.0, 0.0));
    world.create_joint(Joint::Spherical(SphericalJoint::new(
        left,
        right,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
    )));
    for _ in 0..120 {
        world.step(DT);
    }
    for body in [left, right] {
        let y = world.body(body).unwrap().position.y;
        assert!(
            (0.45..=0.55).contains(&y),
            "body {body} should rest on the ground at {y}"
        );
        assert!(touches(&world, ground, body));
    }
}
