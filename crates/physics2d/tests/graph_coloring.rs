//! The constraint graph colors constraints like Box2D v3.2's (P2D-17).
//!
//! Before FIX-2.5 the graph was rebuilt every step with 8 colors, each constraint taking the
//! first color free of its bodies in slot order, and the overflow solved last. Box2D keeps 24
//! colors and gives a constraint its color when it joins the graph, for as long as it stays:
//! constraints between two dynamic bodies take the first free of the low 20 colors, those with a
//! static or kinematic body the last free color below the overflow, so they are solved after
//! the others and have the last word, and the overflow is solved first. That order decides how
//! a resting box takes a hit. Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_5_graph_coloring.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::{Collider2d, Filter2d};
use korppi_physics2d::joint::{Joint2d, RevoluteJoint2d};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d, WorldConfig2d};

const DT: f32 = 1.0 / 60.0;

fn world(sleep: bool) -> PhysicsWorld2d {
    PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -10.0))
        .with_config(WorldConfig2d {
            enable_sleep: sleep,
            ..WorldConfig2d::default()
        })
}

fn add_box(
    world: &mut PhysicsWorld2d,
    kind: BodyType,
    x: f32,
    y: f32,
    hx: f32,
    hy: f32,
) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(kind).with_position(Vec2::new(x, y)));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(hx, hy))).with_friction(0.6),
    );
    id
}

/// Box2D's constraints per color, from the listed (color, count) pairs.
fn colors(counts: &[(usize, usize)]) -> [usize; 24] {
    let mut all = [0; 24];
    for &(color, count) in counts {
        all[color] = count;
    }
    all
}

#[test]
fn a_resting_box_takes_a_hit_like_box2d() {
    // A box falls asleep on the ground at step 30; a ball dropped at step 60 hits it at step 83.
    // The ground contact, in a high color, is solved after the ball's: Box2D v3.2 has the box
    // barely sinking (0.00059 m/s) while the ball stops, and the box asleep again at step 113.
    let mut world = world(true);
    add_box(&mut world, BodyType::Static, 0.0, -0.5, 20.0, 0.5);
    let target = add_box(&mut world, BodyType::Dynamic, 0.0, 0.5, 0.5, 0.5);
    let mut ball = None;
    let mut asleep_again = None;
    for step in 1..=140 {
        if step == 60 {
            let id = world.create_body(
                RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 2.0)),
            );
            world.create_collider(
                id,
                Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25)))
                    .with_friction(0.6),
            );
            ball = Some(id);
        }
        world.step(DT);
        let (b, t) = (
            world.body(ball.unwrap_or(target)).unwrap(),
            world.body(target).unwrap(),
        );
        // Box2D: (step, box y, box vy, ball y, ball vy).
        for (at, box_y, box_vy, ball_y, ball_vy) in [
            (83, 0.499_911, -0.000_590_6, 1.250_462, -0.013_530),
            (84, 0.499_906_6, 0.000_040_2, 1.249_902, -0.000_114),
        ] {
            if step == at {
                assert!(!t.is_sleeping, "step {step}: the box is awake");
                assert!(
                    (t.position.y - box_y).abs() < 2e-6,
                    "step {step}: box y {}",
                    t.position.y
                );
                assert!(
                    (t.linear_velocity.y - box_vy).abs() < 2e-4,
                    "step {step}: box vy {}",
                    t.linear_velocity.y
                );
                assert!(
                    (b.position.y - ball_y).abs() < 1e-4,
                    "step {step}: ball y {}",
                    b.position.y
                );
                assert!(
                    (b.linear_velocity.y - ball_vy).abs() < 2e-3,
                    "step {step}: ball vy {}",
                    b.linear_velocity.y
                );
            }
        }
        if step > 83 && t.is_sleeping && asleep_again.is_none() {
            asleep_again = Some(step);
        }
    }
    assert_eq!(asleep_again, Some(113));
}

#[test]
fn a_pyramid_is_colored_like_box2d() {
    // A ten-row pyramid of unit boxes on the ground: 45 contacts between boxes spread over six
    // colors, and the ten on the ground in color 22, from the first step to the last.
    let mut world = world(false);
    add_box(&mut world, BodyType::Static, 0.0, -0.5, 20.0, 0.5);
    let mut top = None;
    for row in 0..10 {
        for i in 0..10 - row {
            let x = -4.5 + 0.5 * row as f32 + i as f32;
            top = Some(add_box(
                &mut world,
                BodyType::Dynamic,
                x,
                0.5 + row as f32,
                0.5,
                0.5,
            ));
        }
    }
    let box2d = colors(&[
        (0, 27),
        (1, 27),
        (2, 22),
        (3, 25),
        (4, 18),
        (5, 16),
        (22, 10),
    ]);
    world.step(DT);
    assert_eq!(world.counters().color_counts, box2d, "step 1");
    for _ in 2..=120 {
        world.step(DT);
    }
    assert_eq!(world.counters().color_counts, box2d, "step 120");
    let top = world.body(top.unwrap()).unwrap().position;
    assert!(
        (top.x + 0.001_28).abs() < 2e-3 && (top.y - 9.492_35).abs() < 2e-3,
        "top at {top:?}"
    );
}

#[test]
fn constraints_that_fit_no_color_overflow() {
    // Thirty small boxes on one dynamic plank: each contact involves the plank, so the 20
    // dynamic colors take one each and the other ten overflow; the plank's ground contact takes
    // color 22.
    let mut world = world(false);
    add_box(&mut world, BodyType::Static, 0.0, -0.5, 40.0, 0.5);
    let plank = add_box(&mut world, BodyType::Dynamic, 0.0, 0.25, 30.0, 0.25);
    for i in 0..30 {
        add_box(
            &mut world,
            BodyType::Dynamic,
            -29.0 + 2.0 * i as f32,
            0.75,
            0.25,
            0.25,
        );
    }
    let mut box2d = colors(&[(22, 1), (23, 10)]);
    box2d[..20].fill(1);
    world.step(DT);
    assert_eq!(world.counters().color_counts, box2d, "step 1");
    for _ in 2..=60 {
        world.step(DT);
    }
    assert_eq!(world.counters().color_counts, box2d, "step 60");
    let y = world.body(plank).unwrap().position.y;
    assert!((y - 0.249_82).abs() < 1e-4, "plank y {y}");
}

#[test]
fn joints_are_colored_when_created() {
    // Twenty links hanging from a static anchor by hinges: the links' joints alternate between
    // colors 0 and 1, the anchor's joint takes color 22, from creation on.
    let mut world = world(false);
    let mut previous =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 20.0)));
    for i in 0..20 {
        let link = world.create_body(
            RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.5 + i as f32, 20.0)),
        );
        world.create_collider(
            link,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.1))).with_filter(
                Filter2d {
                    group_index: -1,
                    ..Filter2d::DEFAULT
                },
            ),
        );
        let anchor_a = if i == 0 {
            Vec2::ZERO
        } else {
            Vec2::new(0.5, 0.0)
        };
        world.create_joint(
            previous,
            link,
            Joint2d::Revolute(RevoluteJoint2d::new(
                anchor_a,
                Vec2::new(-0.5, 0.0),
                0.0,
                0.0,
            )),
        );
        previous = link;
    }
    let box2d = colors(&[(0, 10), (1, 9), (22, 1)]);
    assert_eq!(world.counters().color_counts, box2d, "created");
    for _ in 0..120 {
        world.step(DT);
    }
    assert_eq!(world.counters().color_counts, box2d, "step 120");
}
