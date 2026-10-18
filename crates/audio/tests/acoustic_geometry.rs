//! The acoustic scene from the physics world (FIX-7.5, ADR-051; finding AUD-08).
//!
//! The owner decided that raytraced mode traces the level's physics colliders. An
//! `AcousticGeometry` is a snapshot of the static ones (shape, world pose and acoustic material)
//! in a bounding volume hierarchy, which the acoustics worker can read while physics steps. Before
//! FIX-7.5 nothing built a scene from level geometry, and `AcousticScene` tested every triangle.

use korppi_audio::raytrace::{AcousticGeometry, AcousticMaterial, AcousticRay};
use korppi_audio::spatial::AcousticRaycaster;
use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, ColliderId3d, Mesh3d, PhysicsWorld, RigidBody, Shape};

/// A body of `body_type` at `position`, turned by `rotation`, with one collider of `shape`.
fn add(
    world: &mut PhysicsWorld,
    body_type: BodyType,
    position: Vec3,
    rotation: Quat,
    shape: Shape,
) -> ColliderId3d {
    let body = world.create_body(RigidBody {
        body_type,
        position,
        rotation,
        ..RigidBody::default()
    });
    world.create_collider(body, Collider::new(shape))
}

fn wall(half_extents: Vec3) -> Shape {
    Shape::Box { half_extents }
}

/// Static colliders are the scene, with their materials; dynamic ones are not.
#[test]
fn static_colliders_become_the_acoustic_scene() {
    let mut world = PhysicsWorld::new();
    let concrete = add(
        &mut world,
        BodyType::Static,
        Vec3::new(0.0, 0.0, -5.0),
        Quat::IDENTITY,
        wall(Vec3::new(5.0, 5.0, 0.25)),
    );
    add(
        &mut world,
        BodyType::Dynamic,
        Vec3::new(0.0, 0.0, -2.0),
        Quat::IDENTITY,
        wall(Vec3::splat(0.5)),
    );
    let geometry = AcousticGeometry::from_physics(&world, |collider| {
        (collider == concrete).then_some(AcousticMaterial::concrete())
    });
    assert_eq!(geometry.surface_count(), 1, "only the static wall");
    let hit = geometry
        .raycast_closest(&AcousticRay::new(Vec3::ZERO, Vec3::NEG_Z), 100.0)
        .expect("the wall");
    assert!(
        (hit.distance - 4.75).abs() < 1e-4,
        "the near face: {}",
        hit.distance
    );
    assert!(
        (hit.normal - Vec3::Z).length() < 1e-4,
        "facing the ray: {:?}",
        hit.normal
    );
    assert_eq!(hit.material, AcousticMaterial::concrete());
}

/// A collider's pose is its body's pose and the collider's local frame.
#[test]
fn bodies_place_their_colliders() {
    let mut world = PhysicsWorld::new();
    // A 4 m wide, 0.5 m thick wall turned 90° about +Y: it spans z and is thin along x.
    add(
        &mut world,
        BodyType::Static,
        Vec3::new(3.0, 0.0, 0.0),
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        wall(Vec3::new(2.0, 2.0, 0.25)),
    );
    let geometry = AcousticGeometry::from_physics(&world, |_| None);
    let hit = geometry
        .raycast_closest(&AcousticRay::new(Vec3::ZERO, Vec3::X), 100.0)
        .expect("the turned wall");
    assert!((hit.distance - 2.75).abs() < 1e-4, "{}", hit.distance);
    assert!(
        geometry
            .raycast_closest(&AcousticRay::new(Vec3::new(0.0, 0.0, 3.0), Vec3::X), 100.0)
            .is_none(),
        "it is 4 m wide along z, not along x"
    );
}

/// Triangle meshes are traced too, and a collider with no material gets the default.
#[test]
fn triangle_meshes_and_the_default_material() {
    let mut world = PhysicsWorld::new();
    let quad = Mesh3d::new(
        &[
            Vec3::new(-2.0, -2.0, 0.0),
            Vec3::new(2.0, -2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(-2.0, 2.0, 0.0),
        ],
        &[[0, 1, 2], [0, 2, 3]],
    )
    .expect("a quad");
    add(
        &mut world,
        BodyType::Static,
        Vec3::new(0.0, 0.0, -3.0),
        Quat::IDENTITY,
        Shape::TriangleMesh(quad),
    );
    let geometry = AcousticGeometry::from_physics(&world, |_| None);
    let hit = geometry
        .raycast_closest(&AcousticRay::new(Vec3::ZERO, Vec3::NEG_Z), 100.0)
        .expect("the quad");
    assert!((hit.distance - 3.0).abs() < 1e-4);
    assert_eq!(hit.material, AcousticMaterial::default());
}

/// Transmission multiplies each wall's per band once (a solid wall has two faces).
#[test]
fn transmission_counts_each_wall_once() {
    let mut world = PhysicsWorld::new();
    let first = add(
        &mut world,
        BodyType::Static,
        Vec3::new(0.0, 0.0, -3.0),
        Quat::IDENTITY,
        wall(Vec3::new(5.0, 5.0, 0.1)),
    );
    add(
        &mut world,
        BodyType::Static,
        Vec3::new(0.0, 0.0, -6.0),
        Quat::IDENTITY,
        wall(Vec3::new(5.0, 5.0, 0.1)),
    );
    let geometry = AcousticGeometry::from_physics(&world, |collider| {
        Some(if collider == first {
            AcousticMaterial::glass()
        } else {
            AcousticMaterial::drywall()
        })
    });
    let ray = AcousticRay::new(Vec3::ZERO, Vec3::NEG_Z);
    let (glass, drywall) = (
        AcousticMaterial::glass().transmission,
        AcousticMaterial::drywall().transmission,
    );
    let both = geometry.raycast_transmission(&ray, 10.0);
    let only_first = geometry.raycast_transmission(&ray, 4.5);
    for band in 0..3 {
        assert!(
            (both[band] - glass[band] * drywall[band]).abs() < 1e-6,
            "band {band}: {both:?}"
        );
        assert!(
            (only_first[band] - glass[band]).abs() < 1e-6,
            "band {band}: {only_first:?}"
        );
    }
    assert_eq!(
        geometry.raycast_transmission(&ray, 2.0),
        [1.0; 3],
        "before the walls"
    );
}

/// Occlusion queries see the geometry: a blocked line of sight is occluded, a clear one not.
#[test]
fn occlusion_follows_the_geometry() {
    let mut world = PhysicsWorld::new();
    add(
        &mut world,
        BodyType::Static,
        Vec3::new(0.0, 0.0, -5.0),
        Quat::IDENTITY,
        wall(Vec3::new(2.0, 2.0, 0.25)),
    );
    let geometry = AcousticGeometry::from_physics(&world, |_| Some(AcousticMaterial::concrete()));
    let occluded = geometry.raycast_occlusion(Vec3::ZERO, Vec3::new(0.0, 0.0, -10.0));
    assert!(
        occluded.is_some_and(|o| o > 0.5),
        "behind the wall: {occluded:?}"
    );
    let clear = geometry.raycast_occlusion(Vec3::ZERO, Vec3::new(5.0, 0.0, -10.0));
    assert_eq!(clear, None, "past its edge");
}

/// A triangle mesh is crossed once per layer: two parallel quads in one mesh transmit the
/// material's transmission squared, and one quad once.
#[test]
fn mesh_layers_each_transmit() {
    let quad = |z: f32| {
        [
            Vec3::new(-2.0, -2.0, z),
            Vec3::new(2.0, -2.0, z),
            Vec3::new(2.0, 2.0, z),
            Vec3::new(-2.0, 2.0, z),
        ]
    };
    let vertices: Vec<Vec3> = quad(-3.0).into_iter().chain(quad(-4.0)).collect();
    let mesh = Mesh3d::new(&vertices, &[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]).unwrap();
    let mut world = PhysicsWorld::new();
    add(
        &mut world,
        BodyType::Static,
        Vec3::ZERO,
        Quat::IDENTITY,
        Shape::TriangleMesh(mesh),
    );
    let geometry = AcousticGeometry::from_physics(&world, |_| Some(AcousticMaterial::glass()));
    let ray = AcousticRay::new(Vec3::ZERO, Vec3::NEG_Z);
    let glass = AcousticMaterial::glass().transmission;
    let both = geometry.raycast_transmission(&ray, 10.0);
    let one = geometry.raycast_transmission(&ray, 3.5);
    for band in 0..3 {
        assert!(
            (both[band] - glass[band] * glass[band]).abs() < 1e-6,
            "{both:?}"
        );
        assert!((one[band] - glass[band]).abs() < 1e-6, "{one:?}");
    }
}
