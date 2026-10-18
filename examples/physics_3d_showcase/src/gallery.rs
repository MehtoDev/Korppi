//! 3D physics demonstration gallery scenes highlighting solver stability, constraints, and spatial queries.

use korppi_math::{Quat, Vec3};
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::joint::{Joint, RevoluteJoint, SphericalJoint};
use korppi_physics::shape::Shape;
use korppi_physics::PhysicsWorld;

/// Common interface for 3D physics gallery demonstration scenes.
pub trait Gallery3d: Send + Sync {
    /// Human-readable title of the gallery.
    fn name(&self) -> &'static str;
    /// Detailed description of the mechanics demonstrated.
    fn description(&self) -> &'static str;
    /// Populates the physics simulation world with the scene's bodies, colliders, and constraints.
    fn setup(&self, world: &mut PhysicsWorld);
}

// ---------------------------------------------------------------------------
// Gallery 1: Heavy Multi-Shape Stacking (100+ Bodies)
// ---------------------------------------------------------------------------
/// Demonstrates compressive mass stability, 100+ active contacts, and island sleep.
pub struct HeavyStackingGallery;

impl Gallery3d for HeavyStackingGallery {
    fn name(&self) -> &'static str {
        "1. Heavy Multi-Shape Stacking (100+ Bodies)"
    }

    fn description(&self) -> &'static str {
        "Cross-hatch Jenga tower, 6-tier 3D pyramid (91 boxes), and mixed shape columns (120+ bodies)"
    }

    fn setup(&self, world: &mut PhysicsWorld) {
        // Ground floor
        let ground = world.create_body(RigidBody {
            body_type: BodyType::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        });
        world.create_collider(
            ground,
            Collider::new(Shape::Box {
                half_extents: Vec3::new(30.0, 0.5, 30.0),
            })
            .with_friction(0.6),
        );

        // 1. 3D Jenga / Cross-Hatch Tower (30 boxes) at x = -10.0, z = 0.0
        let jenga_hw = 0.4;
        let jenga_hh = 0.25;
        let jenga_hl = 1.3;

        for layer in 0..10 {
            let y = layer as f32 * (jenga_hh * 2.0 + 0.02) + jenga_hh;
            let is_z_axis = layer % 2 == 1;

            for slot in 0..3 {
                let offset = (slot as f32 - 1.0) * (jenga_hw * 2.1);
                let (pos, half_extents) = if is_z_axis {
                    (
                        Vec3::new(-10.0 + offset, y, 0.0),
                        Vec3::new(jenga_hw, jenga_hh, jenga_hl),
                    )
                } else {
                    (
                        Vec3::new(-10.0, y, offset),
                        Vec3::new(jenga_hl, jenga_hh, jenga_hw),
                    )
                };

                let b = world.create_body(RigidBody {
                    body_type: BodyType::Dynamic,
                    position: pos,
                    mass: 1.0,
                    ..Default::default()
                });
                world.create_collider(
                    b,
                    Collider::new(Shape::Box { half_extents }).with_friction(0.6),
                );
            }
        }

        // 2. 6-Tier 3D Pyramid (36 + 25 + 16 + 9 + 4 + 1 = 91 boxes) at x = 2.0, z = 0.0
        let box_half = 0.45;
        let step_size = box_half * 2.05;
        let mut cur_y = box_half;

        for tier in (1..=6).rev() {
            let count = tier;
            let start_coord = -(count as f32 - 1.0) * 0.5 * step_size;

            for r in 0..count {
                for c in 0..count {
                    let px = 2.0 + start_coord + r as f32 * step_size;
                    let pz = start_coord + c as f32 * step_size;

                    let b = world.create_body(RigidBody {
                        body_type: BodyType::Dynamic,
                        position: Vec3::new(px, cur_y, pz),
                        mass: 1.5,
                        ..Default::default()
                    });
                    world.create_collider(
                        b,
                        Collider::new(Shape::Box {
                            half_extents: Vec3::splat(box_half),
                        })
                        .with_friction(0.5),
                    );
                }
            }
            cur_y += box_half * 2.02;
        }

        // 3. Multi-shape column at x = 12.0 (spheres, capsules, boxes)
        for i in 0..12 {
            let y = i as f32 * 1.1 + 0.6;
            let b = world.create_body(RigidBody {
                body_type: BodyType::Dynamic,
                position: Vec3::new(12.0, y, 0.0),
                mass: 1.0,
                ..Default::default()
            });

            let shape = match i % 3 {
                0 => Shape::Box {
                    half_extents: Vec3::splat(0.4),
                },
                1 => Shape::Sphere { radius: 0.4 },
                _ => Shape::Capsule {
                    radius: 0.3,
                    half_height: 0.25,
                },
            };

            world.create_collider(b, Collider::new(shape).with_friction(0.5));
        }
    }
}

// ---------------------------------------------------------------------------
// Gallery 2: Soft-Step Solver Live Interactive Tuning
// ---------------------------------------------------------------------------
/// Demonstrates soft compliant spring versus stiff dynamics on dual towers.
pub struct SoftStepTuningGallery;

impl Gallery3d for SoftStepTuningGallery {
    fn name(&self) -> &'static str {
        "2. Soft-Step Solver Live Tuning"
    }

    fn description(&self) -> &'static str {
        "Dual side-by-side towers comparing compliant soft-spring stacking vs stiff rigid dynamics"
    }

    fn setup(&self, world: &mut PhysicsWorld) {
        // Ground floor
        let ground = world.create_body(RigidBody {
            body_type: BodyType::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        });
        world.create_collider(
            ground,
            Collider::new(Shape::Box {
                half_extents: Vec3::new(25.0, 0.5, 25.0),
            })
            .with_friction(0.7),
        );

        // Left Tower: Default Stacking (x = -4.0)
        let box_h = 0.5;
        for i in 0..12 {
            let y = i as f32 * 1.05 + box_h;
            let b = world.create_body(RigidBody {
                body_type: BodyType::Dynamic,
                position: Vec3::new(-4.0, y, 0.0),
                mass: 2.0,
                ..Default::default()
            });
            world.create_collider(
                b,
                Collider::new(Shape::Box {
                    half_extents: Vec3::splat(box_h),
                })
                .with_friction(0.6),
            );
        }

        // Right Tower: Compliant Stacking (x = 4.0)
        for i in 0..12 {
            let y = i as f32 * 1.05 + box_h;
            let b = world.create_body(RigidBody {
                body_type: BodyType::Dynamic,
                position: Vec3::new(4.0, y, 0.0),
                mass: 2.0,
                ..Default::default()
            });
            world.create_collider(
                b,
                Collider::new(Shape::Box {
                    half_extents: Vec3::splat(box_h),
                })
                .with_friction(0.6),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Gallery 3: Friction & Restitution Testing Ramps
// ---------------------------------------------------------------------------
/// Demonstrates Coulomb friction sliding ramps and restitution bouncing spheres.
pub struct FrictionRestitutionGallery;

impl Gallery3d for FrictionRestitutionGallery {
    fn name(&self) -> &'static str {
        "3. Friction & Restitution Testing Ramps"
    }

    fn description(&self) -> &'static str {
        "Inclined ramps (mu = 0.0, 0.2, 0.5, 1.0) and restitution drop tests (e = 0.0, 0.5, 1.0)"
    }

    fn setup(&self, world: &mut PhysicsWorld) {
        // Ground floor
        let ground = world.create_body(RigidBody {
            body_type: BodyType::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        });
        world.create_collider(
            ground,
            Collider::new(Shape::Box {
                half_extents: Vec3::new(25.0, 0.5, 25.0),
            })
            .with_friction(0.5)
            .with_restitution(0.0),
        );

        // 1. Inclined Ramps (30 degrees around Z axis)
        let ramp_angle = -30.0f32.to_radians();
        let ramp_rot = Quat::from_rotation_z(ramp_angle);

        let frictions = [0.0f32, 0.2, 0.5, 1.0];
        for (i, &mu) in frictions.iter().enumerate() {
            let z_pos = -9.0 + i as f32 * 6.0;

            // Static Ramp
            let ramp = world.create_body(RigidBody {
                body_type: BodyType::Static,
                position: Vec3::new(-4.0, 5.0, z_pos),
                rotation: ramp_rot,
                ..Default::default()
            });
            world.create_collider(
                ramp,
                Collider::new(Shape::Box {
                    half_extents: Vec3::new(4.5, 0.15, 1.2),
                })
                .with_friction(mu),
            );

            // Sliding Box on ramp
            let box_body = world.create_body(RigidBody {
                body_type: BodyType::Dynamic,
                position: Vec3::new(-6.0, 6.4, z_pos - 0.5),
                rotation: ramp_rot,
                mass: 1.0,
                ..Default::default()
            });
            world.create_collider(
                box_body,
                Collider::new(Shape::Box {
                    half_extents: Vec3::splat(0.3),
                })
                .with_friction(mu),
            );

            // Rolling Sphere on ramp
            let sphere_body = world.create_body(RigidBody {
                body_type: BodyType::Dynamic,
                position: Vec3::new(-6.0, 6.5, z_pos + 0.5),
                rotation: ramp_rot,
                mass: 1.0,
                ..Default::default()
            });
            world.create_collider(
                sphere_body,
                Collider::new(Shape::Sphere { radius: 0.3 }).with_friction(mu),
            );
        }

        // 2. Bouncy Floor Restitution Drops at x = 8.0
        let restitutions = [0.0f32, 0.5, 1.0];
        for (i, &rest) in restitutions.iter().enumerate() {
            let z = -4.0 + i as f32 * 4.0;
            let b = world.create_body(RigidBody {
                body_type: BodyType::Dynamic,
                position: Vec3::new(8.0, 10.0, z),
                mass: 1.0,
                ..Default::default()
            });
            world.create_collider(
                b,
                Collider::new(Shape::Sphere { radius: 0.5 }).with_restitution(rest),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Gallery 4: Dynamic 3D Ragdoll Simulation
// ---------------------------------------------------------------------------
/// Demonstrates articulated 3D humanoid skeleton tumbling down stairs and ramps.
pub struct Ragdoll3dGallery;

impl Gallery3d for Ragdoll3dGallery {
    fn name(&self) -> &'static str {
        "4. Dynamic 3D Ragdoll Simulation"
    }

    fn description(&self) -> &'static str {
        "Articulated humanoid ragdolls with cone and hinge limits tumbling through obstacles"
    }

    fn setup(&self, world: &mut PhysicsWorld) {
        // Ground floor
        let ground = world.create_body(RigidBody {
            body_type: BodyType::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        });
        world.create_collider(
            ground,
            Collider::new(Shape::Box {
                half_extents: Vec3::new(25.0, 0.5, 25.0),
            })
            .with_friction(0.6),
        );

        // Staircase obstacle (5 steps) at x = -8.0 to -3.0
        for i in 0..5 {
            let x = -8.0 + i as f32 * 1.0;
            let y = 0.4 + i as f32 * 0.7;
            let step = world.create_body(RigidBody {
                body_type: BodyType::Static,
                position: Vec3::new(x, y, 0.0),
                ..Default::default()
            });
            world.create_collider(
                step,
                Collider::new(Shape::Box {
                    half_extents: Vec3::new(0.6, 0.35, 3.0),
                })
                .with_friction(0.5),
            );
        }

        // Inclined tumble ramp at x = 2.0 to 8.0
        let ramp = world.create_body(RigidBody {
            body_type: BodyType::Static,
            position: Vec3::new(5.0, 4.0, 0.0),
            rotation: Quat::from_rotation_z(-25.0f32.to_radians()),
            ..Default::default()
        });
        world.create_collider(
            ramp,
            Collider::new(Shape::Box {
                half_extents: Vec3::new(4.0, 0.2, 3.0),
            })
            .with_friction(0.4),
        );

        // Spawn Ragdoll 1 above stairs
        spawn_ragdoll_3d(world, Vec3::new(-6.0, 8.0, 0.0));

        // Spawn Ragdoll 2 above ramp
        spawn_ragdoll_3d(world, Vec3::new(3.0, 9.0, 0.0));
    }
}

/// Spawns an articulated 3D humanoid ragdoll with anatomical cone and hinge limits.
fn spawn_ragdoll_3d(world: &mut PhysicsWorld, root: Vec3) {
    // 1. Pelvis
    let pelvis = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(0.0, 0.0, 0.0),
        mass: 5.0,
        ..Default::default()
    });
    world.create_collider(
        pelvis,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.22, 0.15, 0.18),
        }),
    );

    // 2. Torso / Chest
    let torso = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(0.0, 0.55, 0.0),
        mass: 8.0,
        ..Default::default()
    });
    world.create_collider(
        torso,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.25, 0.35, 0.2),
        }),
    );
    let spine = SphericalJoint::new(
        pelvis,
        torso,
        Vec3::new(0.0, 0.15, 0.0),
        Vec3::new(0.0, -0.35, 0.0),
    )
    .with_cone_limit(20.0f32.to_radians());
    world.create_joint(Joint::Spherical(spine));

    // 3. Head
    let head = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(0.0, 1.15, 0.0),
        mass: 3.0,
        ..Default::default()
    });
    world.create_collider(head, Collider::new(Shape::Sphere { radius: 0.22 }));
    let neck = SphericalJoint::new(
        torso,
        head,
        Vec3::new(0.0, 0.35, 0.0),
        Vec3::new(0.0, -0.22, 0.0),
    )
    .with_cone_limit(25.0f32.to_radians());
    world.create_joint(Joint::Spherical(neck));

    // 4. Left Arm (Upper arm + Forearm)
    let l_upper_arm = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(-0.55, 0.7, 0.0),
        mass: 2.0,
        ..Default::default()
    });
    world.create_collider(
        l_upper_arm,
        Collider::new(Shape::Capsule {
            radius: 0.09,
            half_height: 0.2,
        }),
    );
    let l_shoulder = SphericalJoint::new(
        torso,
        l_upper_arm,
        Vec3::new(-0.25, 0.25, 0.0),
        Vec3::new(0.0, 0.2, 0.0),
    )
    .with_cone_limit(75.0f32.to_radians());
    world.create_joint(Joint::Spherical(l_shoulder));

    let l_forearm = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(-0.55, 0.2, 0.0),
        mass: 1.5,
        ..Default::default()
    });
    world.create_collider(
        l_forearm,
        Collider::new(Shape::Capsule {
            radius: 0.07,
            half_height: 0.2,
        }),
    );
    let l_elbow = RevoluteJoint::new(
        l_upper_arm,
        l_forearm,
        Vec3::new(0.0, -0.2, 0.0),
        Vec3::new(0.0, 0.2, 0.0),
        Vec3::X,
    )
    .with_limits(0.0, 130.0f32.to_radians());
    world.create_joint(Joint::Revolute(l_elbow));

    // 5. Right Arm (Upper arm + Forearm)
    let r_upper_arm = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(0.55, 0.7, 0.0),
        mass: 2.0,
        ..Default::default()
    });
    world.create_collider(
        r_upper_arm,
        Collider::new(Shape::Capsule {
            radius: 0.09,
            half_height: 0.2,
        }),
    );
    let r_shoulder = SphericalJoint::new(
        torso,
        r_upper_arm,
        Vec3::new(0.25, 0.25, 0.0),
        Vec3::new(0.0, 0.2, 0.0),
    )
    .with_cone_limit(75.0f32.to_radians());
    world.create_joint(Joint::Spherical(r_shoulder));

    let r_forearm = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(0.55, 0.2, 0.0),
        mass: 1.5,
        ..Default::default()
    });
    world.create_collider(
        r_forearm,
        Collider::new(Shape::Capsule {
            radius: 0.07,
            half_height: 0.2,
        }),
    );
    let r_elbow = RevoluteJoint::new(
        r_upper_arm,
        r_forearm,
        Vec3::new(0.0, -0.2, 0.0),
        Vec3::new(0.0, 0.2, 0.0),
        Vec3::X,
    )
    .with_limits(-130.0f32.to_radians(), 0.0);
    world.create_joint(Joint::Revolute(r_elbow));

    // 6. Left Leg (Thigh + Calf)
    let l_thigh = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(-0.2, -0.5, 0.0),
        mass: 4.0,
        ..Default::default()
    });
    world.create_collider(
        l_thigh,
        Collider::new(Shape::Capsule {
            radius: 0.11,
            half_height: 0.25,
        }),
    );
    let l_hip = SphericalJoint::new(
        pelvis,
        l_thigh,
        Vec3::new(-0.15, -0.15, 0.0),
        Vec3::new(0.0, 0.25, 0.0),
    )
    .with_cone_limit(60.0f32.to_radians());
    world.create_joint(Joint::Spherical(l_hip));

    let l_calf = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(-0.2, -1.1, 0.0),
        mass: 2.5,
        ..Default::default()
    });
    world.create_collider(
        l_calf,
        Collider::new(Shape::Capsule {
            radius: 0.09,
            half_height: 0.25,
        }),
    );
    let l_knee = RevoluteJoint::new(
        l_thigh,
        l_calf,
        Vec3::new(0.0, -0.25, 0.0),
        Vec3::new(0.0, 0.25, 0.0),
        Vec3::X,
    )
    .with_limits(-130.0f32.to_radians(), 0.0);
    world.create_joint(Joint::Revolute(l_knee));

    // 7. Right Leg (Thigh + Calf)
    let r_thigh = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(0.2, -0.5, 0.0),
        mass: 4.0,
        ..Default::default()
    });
    world.create_collider(
        r_thigh,
        Collider::new(Shape::Capsule {
            radius: 0.11,
            half_height: 0.25,
        }),
    );
    let r_hip = SphericalJoint::new(
        pelvis,
        r_thigh,
        Vec3::new(0.15, -0.15, 0.0),
        Vec3::new(0.0, 0.25, 0.0),
    )
    .with_cone_limit(60.0f32.to_radians());
    world.create_joint(Joint::Spherical(r_hip));

    let r_calf = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: root + Vec3::new(0.2, -1.1, 0.0),
        mass: 2.5,
        ..Default::default()
    });
    world.create_collider(
        r_calf,
        Collider::new(Shape::Capsule {
            radius: 0.09,
            half_height: 0.25,
        }),
    );
    let r_knee = RevoluteJoint::new(
        r_thigh,
        r_calf,
        Vec3::new(0.0, -0.25, 0.0),
        Vec3::new(0.0, 0.25, 0.0),
        Vec3::X,
    )
    .with_limits(-130.0f32.to_radians(), 0.0);
    world.create_joint(Joint::Revolute(r_knee));
}

// ---------------------------------------------------------------------------
// Gallery 5: Spatial Queries & Interactive Debug Visualization
// ---------------------------------------------------------------------------
/// Demonstrates BVH hierarchy, raycasting intersection tests, and volume overlap queries.
pub struct SpatialQueriesGallery;

impl Gallery3d for SpatialQueriesGallery {
    fn name(&self) -> &'static str {
        "5. Spatial Queries & BVH Debug"
    }

    fn description(&self) -> &'static str {
        "Continuous camera raycast pointer, volume AABB overlap query, and dynamic BVH tree nodes"
    }

    fn setup(&self, world: &mut PhysicsWorld) {
        // Ground floor
        let ground = world.create_body(RigidBody {
            body_type: BodyType::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        });
        world.create_collider(
            ground,
            Collider::new(Shape::Box {
                half_extents: Vec3::new(25.0, 0.5, 25.0),
            }),
        );

        // Grid of test target shapes
        for x_idx in 0..5 {
            for z_idx in 0..5 {
                let px = -8.0 + x_idx as f32 * 4.0;
                let pz = -8.0 + z_idx as f32 * 4.0;

                let b = world.create_body(RigidBody {
                    body_type: BodyType::Dynamic,
                    position: Vec3::new(px, 1.5, pz),
                    mass: 1.0,
                    ..Default::default()
                });

                let shape = match (x_idx + z_idx) % 4 {
                    0 => Shape::Sphere { radius: 0.6 },
                    1 => Shape::Box {
                        half_extents: Vec3::new(0.5, 0.6, 0.5),
                    },
                    2 => Shape::Capsule {
                        radius: 0.4,
                        half_height: 0.5,
                    },
                    _ => Shape::convex_hull(&[
                        Vec3::new(-0.5, -0.5, -0.5),
                        Vec3::new(0.5, -0.5, -0.5),
                        Vec3::new(0.5, -0.5, 0.5),
                        Vec3::new(-0.5, -0.5, 0.5),
                        Vec3::new(0.0, 0.8, 0.0), // Pyramid apex
                    ])
                    .expect("a pyramid has volume"),
                };

                world.create_collider(b, Collider::new(shape));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gallery 6: Bitwise Determinism Verification & State Snapshot Rollback
// ---------------------------------------------------------------------------
/// Demonstrates exact bitwise state snapshots and deterministic simulation rollback.
pub struct RollbackDeterminismGallery;

impl Gallery3d for RollbackDeterminismGallery {
    fn name(&self) -> &'static str {
        "6. Snapshot Determinism & Rollback"
    }

    fn description(&self) -> &'static str {
        "Continuous 120-tick snapshot ring buffer with rewind and bit-exact rollback resumption"
    }

    fn setup(&self, world: &mut PhysicsWorld) {
        // Ground floor
        let ground = world.create_body(RigidBody {
            body_type: BodyType::Static,
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        });
        world.create_collider(
            ground,
            Collider::new(Shape::Box {
                half_extents: Vec3::new(25.0, 0.5, 25.0),
            })
            .with_friction(0.5),
        );

        // Inclined guide ramps
        for i in 0..3 {
            let y = 3.0 + i as f32 * 2.5;
            let dir = if i % 2 == 0 { -1.0 } else { 1.0 };
            let ramp = world.create_body(RigidBody {
                body_type: BodyType::Static,
                position: Vec3::new(dir * 2.0, y, 0.0),
                rotation: Quat::from_rotation_z(dir * 18.0f32.to_radians()),
                ..Default::default()
            });
            world.create_collider(
                ramp,
                Collider::new(Shape::Box {
                    half_extents: Vec3::new(3.0, 0.15, 1.5),
                })
                .with_friction(0.4),
            );
        }

        // Cascading dynamic bodies dropped from top
        for i in 0..20 {
            let offset_x = (i % 4) as f32 * 0.4 - 0.6;
            let offset_z = (i / 4) as f32 * 0.4 - 0.8;
            let y = 11.0 + i as f32 * 0.8;

            let b = world.create_body(RigidBody {
                body_type: BodyType::Dynamic,
                position: Vec3::new(offset_x, y, offset_z),
                mass: 1.0,
                ..Default::default()
            });

            if i % 2 == 0 {
                world.create_collider(
                    b,
                    Collider::new(Shape::Sphere { radius: 0.35 }).with_restitution(0.3),
                );
            } else {
                world.create_collider(
                    b,
                    Collider::new(Shape::Box {
                        half_extents: Vec3::splat(0.3),
                    })
                    .with_friction(0.4),
                );
            }
        }
    }
}
