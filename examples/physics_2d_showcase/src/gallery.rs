//! Gallery scene definitions for 2D physics showcases.

use korppi_math::{Rot2, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::chain::ChainDef2d;
use korppi_physics2d::collider::{Collider2d, SurfaceMaterial2d};
use korppi_physics2d::joint::{
    DistanceJoint2d, Joint2d, PrismaticJoint2d, RevoluteJoint2d, WeldJoint2d, WheelJoint2d,
};
use korppi_physics2d::shape::{Capsule2d, Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::PhysicsWorld2d;

/// Common interface for 2D physics gallery demonstration scenes.
pub trait Gallery2d: Send + Sync {
    /// Human-readable title of the gallery.
    fn name(&self) -> &'static str;
    /// Detailed description of the mechanics demonstrated.
    fn description(&self) -> &'static str;
    /// Populates the physics simulation world with the scene's bodies, colliders, and constraints.
    fn setup(&self, world: &mut PhysicsWorld2d);
}

// ---------------------------------------------------------------------------
// Gallery 1: Dynamic Rigid Body Stacking & Stability
// ---------------------------------------------------------------------------
/// Demonstrates resting contact stability, soft-step compliance, and multi-shape stacking.
pub struct StackingGallery;

impl Gallery2d for StackingGallery {
    fn name(&self) -> &'static str {
        "1. Stacking & Stability"
    }

    fn description(&self) -> &'static str {
        "10-box tall tower, 21-box pyramid, irregular convex polygon wall, and multi-shape columns"
    }

    fn setup(&self, world: &mut PhysicsWorld2d) {
        // Ground floor
        let ground = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -1.0)));
        world.create_collider(
            ground,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(25.0, 1.0))),
        );

        // 1. Vertical 10-box stack at x = -8.0
        let box_half = 0.5;
        for i in 0..10 {
            let y = i as f32 * 1.05 + 0.5;
            let b = world
                .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(-8.0, y)));
            world.create_collider(
                b,
                Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(box_half, box_half)))
                    .with_friction(0.6),
            );
        }

        // 2. 6-tier pyramid (21 boxes) at x = 0.0
        let mut row_start_y = 0.5;
        for row in (1..=6).rev() {
            let count = row;
            let start_x = 0.0 - (count as f32 - 1.0) * 0.55;
            for i in 0..count {
                let x = start_x + i as f32 * 1.1;
                let b = world.create_body(
                    RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, row_start_y)),
                );
                world.create_collider(
                    b,
                    Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)))
                        .with_friction(0.5),
                );
            }
            row_start_y += 1.05;
        }

        // 3. Irregular convex polygon stone wall at x = -4.0
        for layer in 0..4 {
            let y = layer as f32 * 1.0 + 0.5;
            // Hexagon stone
            let hex_verts = vec![
                Vec2::new(-0.5, -0.3),
                Vec2::new(0.5, -0.3),
                Vec2::new(0.6, 0.1),
                Vec2::new(0.3, 0.4),
                Vec2::new(-0.3, 0.4),
                Vec2::new(-0.6, 0.1),
            ];
            let b = world
                .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(-4.0, y)));
            world.create_collider(
                b,
                Collider2d::new(Shape2d::Polygon(
                    Polygon2d::from_vertices(&hex_verts, 0.0).expect("a hexagon"),
                ))
                .with_friction(0.7),
            );
        }

        // 4. Mixed column at x = 7.0 (box, capsule, circle, box, capsule, circle)
        let b1 = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(7.0, 0.5)));
        world.create_collider(
            b1,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.6, 0.5))),
        );

        let b2 = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(7.0, 1.6)));
        world.create_collider(
            b2,
            Collider2d::new(Shape2d::Capsule(Capsule2d::new(
                Vec2::new(-0.4, 0.0),
                Vec2::new(0.4, 0.0),
                0.35,
            ))),
        );

        let b3 = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(7.0, 2.7)));
        world.create_collider(
            b3,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
        );

        let b4 = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(7.0, 3.8)));
        world.create_collider(
            b4,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
        );

        let b5 = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(7.0, 4.9)));
        world.create_collider(
            b5,
            Collider2d::new(Shape2d::Capsule(Capsule2d::new(
                Vec2::new(0.0, -0.3),
                Vec2::new(0.0, 0.3),
                0.35,
            ))),
        );

        let b6 = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(7.0, 6.0)));
        world.create_collider(
            b6,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.45))),
        );
    }
}

// ---------------------------------------------------------------------------
// Gallery 2: Mechanical Joint Linkages & Constraints
// ---------------------------------------------------------------------------
/// Demonstrates mechanical joint constraints: Distance, Revolute, Prismatic, Wheel, and Weld.
pub struct JointsGallery;

impl Gallery2d for JointsGallery {
    fn name(&self) -> &'static str {
        "2. Mechanical Joints & Constraints"
    }

    fn description(&self) -> &'static str {
        "Suspension bridge, motorized rotating wheel, chaotic double pendulum, wheel rover, weld beam"
    }

    fn setup(&self, world: &mut PhysicsWorld2d) {
        // Ground floor
        let ground = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -1.0)));
        world.create_collider(
            ground,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(25.0, 1.0))),
        );

        // 1. Suspension bridge / catenary chain across x = -10.0 to -2.0
        let tower_left = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(-10.0, 6.0)));
        let tower_right = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(-2.0, 6.0)));

        let num_planks = 8;
        let mut prev_body = tower_left;
        let mut prev_anchor = Vec2::ZERO;
        let plank_half_w = 0.45;

        for i in 0..num_planks {
            let t = (i as f32 + 0.5) / num_planks as f32;
            let x = -10.0 + t * 8.0;
            let y = 6.0 - (4.0 * t * (1.0 - t)) * 1.5; // catenary sag

            let plank = world
                .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, y)));
            world.create_collider(
                plank,
                Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(plank_half_w, 0.1)))
                    .with_friction(0.6),
            );

            let dj =
                DistanceJoint2d::new(prev_anchor, Vec2::new(-plank_half_w, 0.0), 0.1, 30.0, 0.7);
            world.create_joint(prev_body, plank, Joint2d::Distance(dj));

            prev_body = plank;
            prev_anchor = Vec2::new(plank_half_w, 0.0);
        }
        // Connect last plank to right tower
        let dj_end = DistanceJoint2d::new(prev_anchor, Vec2::ZERO, 0.1, 30.0, 0.7);
        world.create_joint(prev_body, tower_right, Joint2d::Distance(dj_end));

        // 2. Motorized rotating 4-spoke wheel at x = 3.0, y = 6.0
        let hub_base = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(3.0, 6.0)));
        let hub_rot = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(3.0, 6.0)));
        world.create_collider(
            hub_rot,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(1.8, 0.2))),
        );
        let spoke = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(3.0, 6.0)));
        world.create_collider(
            spoke,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.2, 1.8))),
        );
        let weld_spoke = WeldJoint2d::new(Vec2::ZERO, Vec2::ZERO, 0.0, 0.0);
        world.create_joint(hub_rot, spoke, Joint2d::Weld(weld_spoke));

        let rev_motor =
            RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 0.0, 0.0).with_motor(1.5, 300.0);
        world.create_joint(hub_base, hub_rot, Joint2d::Revolute(rev_motor));

        // 3. Chaotic Double Pendulum at x = 9.0, y = 7.0
        let pend_base = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(9.0, 7.0)));
        let pend_arm1 = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(9.7, 7.0)));
        world.create_collider(
            pend_arm1,
            Collider2d::new(Shape2d::Capsule(Capsule2d::new(
                Vec2::new(-0.7, 0.0),
                Vec2::new(0.7, 0.0),
                0.15,
            ))),
        );
        let rev1 = RevoluteJoint2d::new(Vec2::ZERO, Vec2::new(-0.7, 0.0), 0.0, 0.0);
        world.create_joint(pend_base, pend_arm1, Joint2d::Revolute(rev1));

        let pend_arm2 = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(11.1, 7.0)));
        world.create_collider(
            pend_arm2,
            Collider2d::new(Shape2d::Capsule(Capsule2d::new(
                Vec2::new(-0.7, 0.0),
                Vec2::new(0.7, 0.0),
                0.15,
            ))),
        );
        let rev2 = RevoluteJoint2d::new(Vec2::new(0.7, 0.0), Vec2::new(-0.7, 0.0), 0.0, 0.0);
        world.create_joint(pend_arm1, pend_arm2, Joint2d::Revolute(rev2));

        // 4. Prismatic Piston Slider at x = -6.0, y = 2.0
        let slider_base = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(-6.0, 2.0)));
        let slider_piston = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(-6.0, 3.0)));
        world.create_collider(
            slider_piston,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.4, 0.4))),
        );
        let mut prism = PrismaticJoint2d::new(Vec2::ZERO, Vec2::ZERO, Vec2::Y, 0.0, 0.0);
        prism.enable_limit = true;
        prism.lower_translation = 0.0;
        prism.upper_translation = 2.0;
        world.create_joint(slider_base, slider_piston, Joint2d::Prismatic(prism));

        // 5. Wheel suspension rover on ground at x = 7.0
        let chassis = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(7.0, 1.5)));
        world.create_collider(
            chassis,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(1.2, 0.3))),
        );

        let wheel_left = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(6.0, 0.8)));
        world.create_collider(
            wheel_left,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.4))).with_friction(0.9),
        );
        let wheel_j1 = WheelJoint2d::new(Vec2::new(-1.0, -0.3), Vec2::ZERO, Vec2::Y, 15.0, 0.7);
        world.create_joint(chassis, wheel_left, Joint2d::Wheel(wheel_j1));

        let wheel_right = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(8.0, 0.8)));
        world.create_collider(
            wheel_right,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.4))).with_friction(0.9),
        );
        let wheel_j2 = WheelJoint2d::new(Vec2::new(1.0, -0.3), Vec2::ZERO, Vec2::Y, 15.0, 0.7);
        world.create_joint(chassis, wheel_right, Joint2d::Wheel(wheel_j2));

        // 6. Weld Cantilever Beam at x = -1.0, y = 2.0
        let beam_root = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(-1.0, 2.0)));
        let beam_mid = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(-0.25, 2.0)));
        world.create_collider(
            beam_mid,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.75, 0.1))),
        );
        let weld1 = WeldJoint2d::new(Vec2::ZERO, Vec2::new(-0.75, 0.0), 0.0, 0.0);
        world.create_joint(beam_root, beam_mid, Joint2d::Weld(weld1));

        let beam_tip = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(1.25, 2.0)));
        world.create_collider(
            beam_tip,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.75, 0.1))),
        );
        let weld2 = WeldJoint2d::new(Vec2::new(0.75, 0.0), Vec2::new(-0.75, 0.0), 0.0, 0.0);
        world.create_joint(beam_mid, beam_tip, Joint2d::Weld(weld2));
    }
}

// ---------------------------------------------------------------------------
// Gallery 3: Articulated 2D Ragdoll Physics Simulation
// ---------------------------------------------------------------------------
/// Demonstrates articulated humanoid ragdolls with anatomical angle limits on an obstacle course.
pub struct RagdollGallery;

impl Gallery2d for RagdollGallery {
    fn name(&self) -> &'static str {
        "3. Articulated Humanoid Ragdolls"
    }

    fn description(&self) -> &'static str {
        "Humanoid ragdolls with RevoluteJoint angle limits tumbling through stairs, pegs, and ramps"
    }

    fn setup(&self, world: &mut PhysicsWorld2d) {
        // Ground floor
        let ground = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -1.0)));
        world.create_collider(
            ground,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(25.0, 1.0))),
        );

        // Obstacle 1: Stairs at left (x = -12.0 to -6.0)
        for i in 0..5 {
            let x = -11.0 + i as f32 * 1.0;
            let y = 1.0 + i as f32 * 0.8;
            let step = world
                .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(x, y)));
            world.create_collider(
                step,
                Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.6, 0.2))),
            );
        }

        // Obstacle 2: Inclined ramp in center-left (x = -4.0, y = 5.0, angle = -30 deg)
        let mut ramp_body = RigidBody2d::new(BodyType::Static).with_position(Vec2::new(-3.0, 5.0));
        ramp_body.rotation = Rot2::from_angle(-30.0f32.to_radians());
        let ramp = world.create_body(ramp_body);
        world.create_collider(
            ramp,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(3.5, 0.2))).with_friction(0.3),
        );

        // Obstacle 3: Pinball pegs in center-right (grid of static circles)
        for row in 0..4 {
            for col in 0..4 {
                let x = 2.0 + col as f32 * 1.6 + (row % 2) as f32 * 0.8;
                let y = 2.0 + row as f32 * 1.5;
                let peg = world
                    .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(x, y)));
                world.create_collider(
                    peg,
                    Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25)))
                        .with_restitution(0.5),
                );
            }
        }

        // Spawn Ragdoll 1 above stairs (x = -9.0, y = 8.0)
        spawn_ragdoll(world, Vec2::new(-9.0, 8.0));

        // Spawn Ragdoll 2 above pegs (x = 4.0, y = 9.0)
        spawn_ragdoll(world, Vec2::new(4.0, 9.0));
    }
}

/// Spawns an articulated 11-body humanoid ragdoll connected with limited Revolute joints.
fn spawn_ragdoll(world: &mut PhysicsWorld2d, root: Vec2) {
    // 1. Torso (x, y)
    let torso = world
        .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(0.0, 0.0)));
    world.create_collider(
        torso,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.25, 0.35))).with_friction(0.4),
    );

    // 2. Head (above torso)
    let head = world
        .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(0.0, 0.6)));
    world.create_collider(
        head,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25))),
    );
    let neck = RevoluteJoint2d::new(Vec2::new(0.0, 0.35), Vec2::new(0.0, -0.25), 0.0, 0.0)
        .with_limits(-25.0f32.to_radians(), 25.0f32.to_radians());
    world.create_joint(torso, head, Joint2d::Revolute(neck));

    // 3. Pelvis (below torso)
    let pelvis = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(0.0, -0.53)),
    );
    world.create_collider(
        pelvis,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.22, 0.18))),
    );
    let spine = RevoluteJoint2d::new(Vec2::new(0.0, -0.35), Vec2::new(0.0, 0.18), 0.0, 0.0)
        .with_limits(-20.0f32.to_radians(), 20.0f32.to_radians());
    world.create_joint(torso, pelvis, Joint2d::Revolute(spine));

    // 4. Left Arm (Upper & Forearm)
    let l_upper_arm = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(-0.45, 0.2)),
    );
    world.create_collider(
        l_upper_arm,
        Collider2d::new(Shape2d::Capsule(Capsule2d::new(
            Vec2::new(0.2, 0.0),
            Vec2::new(-0.2, 0.0),
            0.1,
        ))),
    );
    let l_shoulder = RevoluteJoint2d::new(Vec2::new(-0.25, 0.2), Vec2::new(0.2, 0.0), 0.0, 0.0)
        .with_limits(-85.0f32.to_radians(), 85.0f32.to_radians());
    world.create_joint(torso, l_upper_arm, Joint2d::Revolute(l_shoulder));

    let l_forearm = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(-0.85, 0.2)),
    );
    world.create_collider(
        l_forearm,
        Collider2d::new(Shape2d::Capsule(Capsule2d::new(
            Vec2::new(0.2, 0.0),
            Vec2::new(-0.2, 0.0),
            0.08,
        ))),
    );
    let l_elbow = RevoluteJoint2d::new(Vec2::new(-0.2, 0.0), Vec2::new(0.2, 0.0), 0.0, 0.0)
        .with_limits(0.0, 130.0f32.to_radians());
    world.create_joint(l_upper_arm, l_forearm, Joint2d::Revolute(l_elbow));

    // 5. Right Arm (Upper & Forearm)
    let r_upper_arm = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(0.45, 0.2)),
    );
    world.create_collider(
        r_upper_arm,
        Collider2d::new(Shape2d::Capsule(Capsule2d::new(
            Vec2::new(-0.2, 0.0),
            Vec2::new(0.2, 0.0),
            0.1,
        ))),
    );
    let r_shoulder = RevoluteJoint2d::new(Vec2::new(0.25, 0.2), Vec2::new(-0.2, 0.0), 0.0, 0.0)
        .with_limits(-85.0f32.to_radians(), 85.0f32.to_radians());
    world.create_joint(torso, r_upper_arm, Joint2d::Revolute(r_shoulder));

    let r_forearm = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(0.85, 0.2)),
    );
    world.create_collider(
        r_forearm,
        Collider2d::new(Shape2d::Capsule(Capsule2d::new(
            Vec2::new(-0.2, 0.0),
            Vec2::new(0.2, 0.0),
            0.08,
        ))),
    );
    let r_elbow = RevoluteJoint2d::new(Vec2::new(0.2, 0.0), Vec2::new(-0.2, 0.0), 0.0, 0.0)
        .with_limits(-130.0f32.to_radians(), 0.0);
    world.create_joint(r_upper_arm, r_forearm, Joint2d::Revolute(r_elbow));

    // 6. Left Leg (Thigh & Calf)
    let l_thigh = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(-0.15, -0.96)),
    );
    world.create_collider(
        l_thigh,
        Collider2d::new(Shape2d::Capsule(Capsule2d::new(
            Vec2::new(0.0, 0.25),
            Vec2::new(0.0, -0.25),
            0.12,
        ))),
    );
    let l_hip = RevoluteJoint2d::new(Vec2::new(-0.15, -0.18), Vec2::new(0.0, 0.25), 0.0, 0.0)
        .with_limits(-45.0f32.to_radians(), 70.0f32.to_radians());
    world.create_joint(pelvis, l_thigh, Joint2d::Revolute(l_hip));

    let l_calf = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(-0.15, -1.46)),
    );
    world.create_collider(
        l_calf,
        Collider2d::new(Shape2d::Capsule(Capsule2d::new(
            Vec2::new(0.0, 0.25),
            Vec2::new(0.0, -0.25),
            0.1,
        ))),
    );
    let l_knee = RevoluteJoint2d::new(Vec2::new(0.0, -0.25), Vec2::new(0.0, 0.25), 0.0, 0.0)
        .with_limits(-130.0f32.to_radians(), 0.0);
    world.create_joint(l_thigh, l_calf, Joint2d::Revolute(l_knee));

    // 7. Right Leg (Thigh & Calf)
    let r_thigh = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(0.15, -0.96)),
    );
    world.create_collider(
        r_thigh,
        Collider2d::new(Shape2d::Capsule(Capsule2d::new(
            Vec2::new(0.0, 0.25),
            Vec2::new(0.0, -0.25),
            0.12,
        ))),
    );
    let r_hip = RevoluteJoint2d::new(Vec2::new(0.15, -0.18), Vec2::new(0.0, 0.25), 0.0, 0.0)
        .with_limits(-45.0f32.to_radians(), 70.0f32.to_radians());
    world.create_joint(pelvis, r_thigh, Joint2d::Revolute(r_hip));

    let r_calf = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_position(root + Vec2::new(0.15, -1.46)),
    );
    world.create_collider(
        r_calf,
        Collider2d::new(Shape2d::Capsule(Capsule2d::new(
            Vec2::new(0.0, 0.25),
            Vec2::new(0.0, -0.25),
            0.1,
        ))),
    );
    let r_knee = RevoluteJoint2d::new(Vec2::new(0.0, -0.25), Vec2::new(0.0, 0.25), 0.0, 0.0)
        .with_limits(-130.0f32.to_radians(), 0.0);
    world.create_joint(r_thigh, r_calf, Joint2d::Revolute(r_knee));
}

// ---------------------------------------------------------------------------
// Gallery 4: Kinetic Chains & Restitution
// ---------------------------------------------------------------------------
/// Demonstrates momentum propagation in Newton's Cradle, domino tumble, and teeter-totter balance.
pub struct KineticGallery;

impl Gallery2d for KineticGallery {
    fn name(&self) -> &'static str {
        "4. Kinetic Chains & Restitution"
    }

    fn description(&self) -> &'static str {
        "Newton's cradle elastic collisions (e=1.0), 10-domino cascade, and teeter-totter balance"
    }

    fn setup(&self, world: &mut PhysicsWorld2d) {
        // Ground floor
        let ground = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -1.0)));
        world.create_collider(
            ground,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(25.0, 1.0))),
        );

        // 1. Newton's Cradle at x = -7.0
        // Top static beam
        let cradle_rail = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(-7.0, 8.0)));
        let ball_radius = 0.4;
        let string_len = 3.5;

        for i in 0..5 {
            let offset_x = (i as f32 - 2.0) * (ball_radius * 2.0);
            let rail_anchor = Vec2::new(offset_x, 0.0);

            let initial_pos = if i == 0 {
                // Pull the first ball 40 degrees to the left, keeping its string taut
                let pull = 40.0f32.to_radians();
                Vec2::new(
                    -7.0 + offset_x - string_len * pull.sin(),
                    8.0 - string_len * pull.cos(),
                )
            } else {
                Vec2::new(-7.0 + offset_x, 8.0 - string_len)
            };

            let ball =
                world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(initial_pos));
            world.create_collider(
                ball,
                Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, ball_radius)))
                    .with_restitution(1.0)
                    .with_friction(0.0),
            );

            let dj = DistanceJoint2d::new(rail_anchor, Vec2::ZERO, string_len, 60.0, 0.1);
            world.create_joint(cradle_rail, ball, Joint2d::Distance(dj));
        }

        // 2. Domino cascade at x = 0.0 to 6.0
        let domino_half_w = 0.06;
        let domino_half_h = 0.5;
        for i in 0..10 {
            let x = 0.5 + i as f32 * 0.65;
            let mut dom_body =
                RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, domino_half_h));
            if i == 0 {
                // Slightly tilt the first domino so it falls into the second
                dom_body.rotation = Rot2::from_angle(15.0f32.to_radians());
                dom_body.angular_velocity = 2.0;
            }
            let b = world.create_body(dom_body);
            world.create_collider(
                b,
                Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(
                    domino_half_w,
                    domino_half_h,
                )))
                .with_friction(0.5)
                .with_restitution(0.2),
            );
        }

        // 3. Dynamic Balance Scale / Teeter-Totter at x = 8.5
        let pivot = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(8.5, 0.8)));
        let beam = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(8.5, 1.2)));
        world.create_collider(
            beam,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(2.2, 0.15))).with_friction(0.6),
        );
        let teeter_rev = RevoluteJoint2d::new(Vec2::new(0.0, 0.4), Vec2::ZERO, 0.0, 0.0)
            .with_limits(-25.0f32.to_radians(), 25.0f32.to_radians());
        world.create_joint(pivot, beam, Joint2d::Revolute(teeter_rev));

        // Heavy 15 kg box falling on the right side of the beam
        let heavy_box = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(10.0, 4.0)));
        world.create_collider(
            heavy_box,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.4, 0.4)))
                .with_density(15.0 / (0.8 * 0.8)),
        );

        // Light 1 kg box resting on the left side of the beam
        let light_box = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(7.0, 1.6)));
        world.create_collider(
            light_box,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.3, 0.3)))
                .with_density(1.0 / (0.6 * 0.6)),
        );
    }
}

// ---------------------------------------------------------------------------
// Gallery 5: Real-Time 2D Contact & Impulse Visualizer
// ---------------------------------------------------------------------------
/// Demonstrates live contact points, normal impulses, friction ramps, and CCD bullets.
pub struct ContactVisualizerGallery;

impl Gallery2d for ContactVisualizerGallery {
    fn name(&self) -> &'static str {
        "5. Contact & Impulse Visualizer"
    }

    fn description(&self) -> &'static str {
        "Live contact manifold vectors, multi-friction ramps, restitution drops, and high-speed CCD bullets"
    }

    fn setup(&self, world: &mut PhysicsWorld2d) {
        // Ground floor
        let ground = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -1.0)));
        world.create_collider(
            ground,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(25.0, 1.0))),
        );

        // 1. Friction Ramps at x = -12.0 to -2.0
        let ramp_rot = Rot2::from_angle(-22.0f32.to_radians());
        let frictions = [0.0f32, 0.2, 0.5, 0.9];
        for (i, &mu) in frictions.iter().enumerate() {
            let x_base = -12.0 + i as f32 * 3.0;
            let mut ramp_body =
                RigidBody2d::new(BodyType::Static).with_position(Vec2::new(x_base, 5.0));
            ramp_body.rotation = ramp_rot;
            let ramp = world.create_body(ramp_body);
            world.create_collider(
                ramp,
                Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(1.4, 0.1))).with_friction(mu),
            );

            // Sliding box on ramp
            let box_pos = Vec2::new(x_base - 0.7, 5.6);
            let mut slider = RigidBody2d::new(BodyType::Dynamic).with_position(box_pos);
            slider.rotation = ramp_rot;
            let s = world.create_body(slider);
            world.create_collider(
                s,
                Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.25, 0.25)))
                    .with_friction(mu),
            );
        }

        // 2. Restitution comparison balls at x = 1.0, 3.0, 5.0
        let restitutions = [0.0f32, 0.5, 1.0];
        for (i, &rest) in restitutions.iter().enumerate() {
            let x = 1.0 + i as f32 * 2.0;
            let b = world
                .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, 7.0)));
            world.create_collider(
                b,
                Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.4)))
                    .with_restitution(rest),
            );
        }

        // 3. CCD High-Speed Bullet vs 0.05m thin static obstacle at x = 9.0
        let thin_wall = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(9.0, 4.0)));
        world.create_collider(
            thin_wall,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.025, 4.0))),
        );

        // High-speed bullet fired towards wall at 350 m/s
        let mut bullet = RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 4.0));
        bullet.linear_velocity = Vec2::new(350.0, 0.0);
        bullet.set_bullet(true);
        let bullet_body = world.create_body(bullet);
        world.create_collider(
            bullet_body,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.2))).with_restitution(0.3),
        );

        // 4. Hopper / funnel dropping dynamic tumbling pieces
        // Left funnel guide
        let mut left_guide = RigidBody2d::new(BodyType::Static).with_position(Vec2::new(-1.0, 7.5));
        left_guide.rotation = Rot2::from_angle(-45.0f32.to_radians());
        let lg = world.create_body(left_guide);
        world.create_collider(
            lg,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(1.2, 0.1))),
        );

        // Right funnel guide
        let mut right_guide = RigidBody2d::new(BodyType::Static).with_position(Vec2::new(1.0, 7.5));
        right_guide.rotation = Rot2::from_angle(45.0f32.to_radians());
        let rg = world.create_body(right_guide);
        world.create_collider(
            rg,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(1.2, 0.1))),
        );

        // Tumbling particles inside funnel
        for i in 0..4 {
            let offset_x = (i as f32 - 1.5) * 0.3;
            let p = world.create_body(
                RigidBody2d::new(BodyType::Dynamic)
                    .with_position(Vec2::new(offset_x, 9.0 + i as f32 * 0.6)),
            );
            world.create_collider(
                p,
                Collider2d::new(Shape2d::Capsule(Capsule2d::new(
                    Vec2::new(-0.15, 0.0),
                    Vec2::new(0.15, 0.0),
                    0.15,
                ))),
            );
        }

        // Curved bowl under the funnel: a chain running right to left, so that it collides on
        // the inside, with ghost vertices continuing the curve past its ends.
        let steps = 16;
        let bowl_verts: Vec<Vec2> = (0..=steps)
            .map(|i| {
                let t = 1.0 - (i as f32 / steps as f32) * 2.0;
                Vec2::new(t * 2.5, 1.0 + (t * t) * 1.5)
            })
            .collect();
        let ghost1 = bowl_verts[0] * 2.0 - bowl_verts[1];
        let ghost2 = bowl_verts[steps] * 2.0 - bowl_verts[steps - 1];
        let bowl = world.create_body(RigidBody2d::new(BodyType::Static));
        world.create_chain(
            bowl,
            &ChainDef2d::open(bowl_verts, ghost1, ghost2).with_material(SurfaceMaterial2d {
                friction: 0.4,
                ..SurfaceMaterial2d::default()
            }),
        );
    }
}
