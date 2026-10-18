//! The broadphase pairs bodies like Box2D v3.2 (P2D-15).
//!
//! Before FIX-2.5 one tree held every proxy, every awake proxy searched it each step, kinematic
//! bodies paired with static and kinematic ones, and a contact existed only while its colliders
//! touched. As in Box2D, static, kinematic and dynamic proxies live in separate trees, only
//! proxies that left their fat boxes look for new pairs, a pair needs a dynamic body, and a
//! contact lives while the fat boxes overlap, touching or not. Fat boxes reach the speculative
//! distance beyond a shape plus a margin: the speculative distance again for static bodies, an
//! eighth of the shape's size up to 5 cm otherwise. Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_5_broadphase.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d, WorldConfig2d};

const DT: f32 = 1.0 / 60.0;

fn sleepless_world() -> PhysicsWorld2d {
    PhysicsWorld2d::new().with_config(WorldConfig2d {
        enable_sleep: false,
        ..WorldConfig2d::default()
    })
}

fn add_box(
    world: &mut PhysicsWorld2d,
    kind: BodyType,
    position: Vec2,
    hx: f32,
    hy: f32,
) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(kind).with_position(position));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(hx, hy))).with_contact_events(true),
    );
    id
}

#[test]
fn kinematic_bodies_do_not_pair_with_static_or_kinematic_ones() {
    // Box2D v3.2: a kinematic box crossing a static wall and another kinematic box makes no
    // contact and reports nothing.
    let mut world = sleepless_world();
    add_box(&mut world, BodyType::Static, Vec2::new(3.0, 0.0), 0.2, 2.0);
    add_box(
        &mut world,
        BodyType::Kinematic,
        Vec2::new(6.0, 0.0),
        0.5,
        0.5,
    );
    let mover = add_box(&mut world, BodyType::Kinematic, Vec2::ZERO, 0.5, 0.5);
    world.body_mut(mover).unwrap().linear_velocity = Vec2::new(4.0, 0.0);
    let (mut most, mut begins) = (0, 0);
    for _ in 0..180 {
        world.step(DT);
        most = most.max(world.counters().contact_count);
        begins += world.contact_events().begin.len();
    }
    assert_eq!((most, begins), (0, 0));
    assert!((world.body(mover).unwrap().position.x - 12.0).abs() < 0.01);
}

#[test]
fn contacts_exist_while_fat_boxes_overlap() {
    // A column of five unit boxes 0.05 m apart falls onto the ground. Box2D v3.2 keeps four
    // contacts between the boxes from the first step, adds the ground contact by step 20, and
    // counts touching contacts (seen from the boxes, so the four between boxes count twice) as
    // 0, 0, 1, 9, 9, 9, 9 at steps 1, 10, 20, 30, 45, 60 and 120.
    let mut world = sleepless_world();
    add_box(
        &mut world,
        BodyType::Static,
        Vec2::new(0.0, -0.5),
        20.0,
        0.5,
    );
    let boxes: Vec<BodyId2d> = (0..5)
        .map(|i| {
            add_box(
                &mut world,
                BodyType::Dynamic,
                Vec2::new(0.0, 1.0 + 1.05 * i as f32),
                0.5,
                0.5,
            )
        })
        .collect();
    let expected = [
        (1, 4, 0),
        (10, 4, 0),
        (20, 5, 1),
        (30, 5, 9),
        (45, 5, 9),
        (60, 5, 9),
        (120, 5, 9),
    ];
    let mut sample = 0;
    for step in 1..=120 {
        world.step(DT);
        if step == expected[sample].0 {
            let touching: usize = boxes.iter().map(|&b| world.body_contacts(b).count()).sum();
            assert_eq!(
                (world.counters().contact_count, touching),
                (expected[sample].1, expected[sample].2),
                "step {step}: (contacts, touching seen from the boxes)"
            );
            sample += 1;
        }
    }
}

#[test]
fn a_rolling_ball_gains_and_loses_contacts_with_posts_it_never_touches() {
    // A ball rolls past a row of static posts 0.08 m above it: their fat boxes overlap, so
    // contacts come and go though the shapes never touch. Box2D v3.2's contact counts every 5
    // steps from step 5:
    let box2d = [
        1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2,
        2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1,
    ];
    let mut world = sleepless_world();
    add_box(
        &mut world,
        BodyType::Static,
        Vec2::new(0.0, -0.5),
        40.0,
        0.5,
    );
    for i in 0..6 {
        add_box(
            &mut world,
            BodyType::Static,
            Vec2::new(2.0 + 2.0 * i as f32, 1.28),
            0.2,
            0.2,
        );
    }
    let ball = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.0, 0.5))
            .with_linear_velocity(Vec2::new(4.0, 0.0))
            .with_angular_velocity(-8.0),
    );
    world.create_collider(
        ball,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
    );
    let mut counts = Vec::new();
    for step in 1..=210 {
        world.step(DT);
        if step % 5 == 0 {
            counts.push(world.counters().contact_count);
        }
    }
    assert_eq!(counts, box2d);
    assert_eq!(
        world.manifolds().len(),
        1,
        "only the ground touches the ball"
    );
}
