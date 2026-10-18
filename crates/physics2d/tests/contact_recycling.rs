//! Contacts are recycled like Box2D v3.2's.
//!
//! While two bodies have barely moved relative to each other since their contact was last
//! collided (by less than `WorldConfig2d::contact_recycle_distance` at their farthest points,
//! and turned less than about 11.5°), Box2D keeps the manifold and only updates its separations
//! from the motion, as the substeps do; a fast body's contacts are always collided again. With
//! the solver's colors and Box2D's substep state, a stack comes out bit for bit as in Box2D.
//! Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_5_contact_recycling.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d, WorldConfig2d};

const DT: f32 = 1.0 / 60.0;

fn sleepless(recycle_distance: Option<f32>) -> PhysicsWorld2d {
    let mut config = WorldConfig2d {
        enable_sleep: false,
        ..WorldConfig2d::default()
    };
    if let Some(distance) = recycle_distance {
        config.contact_recycle_distance = distance;
    }
    PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -10.0))
        .with_config(config)
}

fn add_box(
    world: &mut PhysicsWorld2d,
    body: RigidBody2d,
    hx: f32,
    hy: f32,
    friction: f32,
) -> BodyId2d {
    let id = world.create_body(body);
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(hx, hy))).with_friction(friction),
    );
    id
}

/// A ten-box stack on the ground run for 2 s: the recycled contacts at steps 1, 2, 3, 4, 10, 60
/// and 120, and the top box's x, y, cosine and sine bits.
fn stack(recycle_distance: Option<f32>, bottom_recycles: bool) -> ([usize; 7], [u32; 4]) {
    let mut world = sleepless(recycle_distance);
    let at = |x: f32, y: f32, kind| RigidBody2d::new(kind).with_position(Vec2::new(x, y));
    add_box(&mut world, at(0.0, -0.5, BodyType::Static), 20.0, 0.5, 0.6);
    let mut top = None;
    for i in 0..10 {
        let body = at(0.0, 0.5 + i as f32, BodyType::Dynamic)
            .with_contact_recycling(i > 0 || bottom_recycles);
        top = Some(add_box(&mut world, body, 0.5, 0.5, 0.6));
    }
    let mut recycled = [0; 7];
    let mut sample = 0;
    for step in 1..=120 {
        world.step(DT);
        if [1, 2, 3, 4, 10, 60, 120].contains(&step) {
            recycled[sample] = world.counters().recycled_contact_count;
            sample += 1;
        }
    }
    let top = world.body(top.unwrap()).unwrap();
    let bits = [
        top.position.x,
        top.position.y,
        top.rotation.c,
        top.rotation.s,
    ]
    .map(f32::to_bits);
    (recycled, bits)
}

#[test]
fn a_resting_stack_recycles_its_contacts_bit_for_bit_like_box2d() {
    // Box2D v3.2: every contact is recycled from the second step on, and the top box ends at
    // x = -0.0018090, y = 9.4674597, exactly.
    let (recycled, bits) = stack(None, true);
    assert_eq!(recycled, [0, 10, 10, 10, 10, 10, 10]);
    assert_eq!(bits, [0xbaed_1a8a, 0x4117_7ab7, 0x3f80_0000, 0x39d1_074c]);
}

#[test]
fn recycling_can_be_switched_off() {
    // Box2D v3.2 with a recycle distance of 0: nothing is recycled, and the top box ends at
    // x = 0.0000955 instead. With the bottom box opting out, its two contacts are collided
    // every step and the other eight recycled.
    let (recycled, bits) = stack(Some(0.0), true);
    assert_eq!(recycled, [0; 7]);
    assert_eq!(bits, [0x38c8_4002, 0x4117_7ab7, 0x3f80_0000, 0x38d6_fb4f]);

    let (recycled, bits) = stack(None, false);
    assert_eq!(recycled, [0, 8, 8, 8, 8, 8, 8]);
    assert_eq!(bits, [0xbaec_8470, 0x4117_7ab8, 0x3f80_0000, 0x39d0_85f2]);
}

#[test]
fn a_sliding_box_is_collided_again_once_it_moved_far_enough() {
    // A box slides from 2 m/s with friction 0.1, 3 cm a step: Box2D v3.2 recycles its ground
    // contact every other step at first (the 5 cm limit is crossed every second step), then
    // more often as it slows.
    let mut world = sleepless(None);
    let at = |x: f32, y: f32, kind| RigidBody2d::new(kind).with_position(Vec2::new(x, y));
    add_box(&mut world, at(0.0, -0.5, BodyType::Static), 20.0, 0.5, 0.1);
    let slider = add_box(
        &mut world,
        at(0.0, 0.5, BodyType::Dynamic).with_linear_velocity(Vec2::new(2.0, 0.0)),
        0.5,
        0.5,
        0.1,
    );
    let mut recycled = Vec::new();
    for _ in 0..40 {
        world.step(DT);
        recycled.push(world.counters().recycled_contact_count);
    }
    let mut box2d = [0, 1].repeat(16);
    box2d.extend([1, 0, 1, 1, 0, 1, 1, 0]);
    assert_eq!(recycled, box2d);
    let body = world.body(slider).unwrap();
    assert!(
        (body.position.x - 1.109_74).abs() < 1e-5,
        "x {}",
        body.position.x
    );
    assert!(
        (body.linear_velocity.x - 1.333_335).abs() < 1e-5,
        "vx {}",
        body.linear_velocity.x
    );
}
