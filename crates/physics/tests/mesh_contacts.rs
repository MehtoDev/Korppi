//! Triangle meshes are Box3D's: a bounding volume hierarchy of sorted triangles with edge
//! flags, one-sided triangles, and contacts that hold one manifold per cluster of triangle
//! normals (P3D-12, P3D-10).
//!
//! `b3CreateMesh` drops degenerate triangles, builds a BVH over the rest (binned SAH, or median
//! splits for grids), sorts the triangles in depth-first order and flags each shared edge as
//! concave, convex or flat (`mesh.c`). A mesh contact queries the BVH around the other shape,
//! collides it with each triangle from the front only (`triangle_manifold.c`), drops the edge
//! and vertex contacts that a neighbouring face contact already covers, so nothing catches on
//! internal edges, and clusters the rest by normal into manifolds (`mesh_contact.c`), separated
//! by a rest offset of one linear slop. Before FIX-3.5 every mesh pair tested every triangle,
//! from both sides, and merged the deepest four points into one manifold.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_5_meshes.c`):
//! 60 Hz, 4 substeps, no sleep, density 1000 kg/m³, static meshes with identified edges.

use std::time::Instant;

use korppi_math::{Quat, Vec3};
use korppi_physics::{
    BodyType, Collider, Mesh3d, MeshEdge, MeshOptions, PhysicsWorld, RigidBody, Shape,
};

const DT: f32 = 1.0 / 60.0;

/// The vertices and triangles of Box3D's `b3CreateGridMesh`, with heights from `height`, split
/// at the median (as Box3D's grids are) or by SAH.
fn grid_mesh(
    (x_count, z_count): (u32, u32),
    cell: f32,
    height: fn(f32, f32) -> f32,
    median_split: bool,
) -> Mesh3d {
    let mut vertices = Vec::new();
    let mut x = -0.5 * cell * x_count as f32;
    for _ in 0..=x_count {
        let mut z = -0.5 * cell * z_count as f32;
        for _ in 0..=z_count {
            vertices.push(Vec3::new(x, height(x, z), z));
            z += cell;
        }
        x += cell;
    }
    let mut indices = Vec::new();
    for ix in 0..x_count {
        for iz in 0..z_count {
            let i1 = iz + (z_count + 1) * ix;
            let i2 = i1 + 1;
            let i3 = i2 + (z_count + 1);
            let i4 = i3 - 1;
            indices.push([i1, i2, i3]);
            indices.push([i3, i4, i1]);
        }
    }
    Mesh3d::with_options(
        &vertices,
        &indices,
        MeshOptions {
            median_split,
            ..MeshOptions::default()
        },
    )
    .expect("a grid makes a mesh")
}

/// A flat n x n grid of `cell` cells, split at the median.
fn grid(n: u32, cell: f32) -> Mesh3d {
    grid_mesh((n, n), cell, |_, _| 0.0, true)
}

/// Waves over a 16 x 16 grid of 0.5 m cells, split by SAH.
fn waves() -> Mesh3d {
    grid_mesh(
        (16, 16),
        0.5,
        |x, z| 0.3 * (0.8 * x).sin() * (0.6 * z).cos(),
        false,
    )
}

/// A V valley: two 2 m slopes at 30 degrees meeting along z at the origin, 2 m deep in z.
fn valley() -> Mesh3d {
    let h = 2.0 * (std::f32::consts::PI / 6.0).tan();
    let vertices = [
        Vec3::new(-2.0, h, -1.0),
        Vec3::new(-2.0, h, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(2.0, h, -1.0),
        Vec3::new(2.0, h, 1.0),
    ];
    let indices = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]];
    Mesh3d::new(&vertices, &indices).expect("the valley makes a mesh")
}

/// The flat, concave and convex edge sides of a mesh.
fn edge_counts(mesh: &Mesh3d) -> (usize, usize, usize) {
    let mut counts = (0, 0, 0);
    for t in 0..mesh.triangle_count() {
        for e in 0..3 {
            match mesh.triangle(t).edge(e) {
                MeshEdge::Flat => counts.0 += 1,
                MeshEdge::Concave => counts.1 += 1,
                MeshEdge::Convex => counts.2 += 1,
                MeshEdge::Open => {}
            }
        }
    }
    counts
}

#[test]
fn meshes_build_box3d_trees() {
    // Box3D: triangle, node, height and degenerate counts, surface area, bounds, edge flags
    // and the first and last triangles in the BVH's depth-first order.
    let cases = [
        (
            "grid",
            grid(8, 1.0),
            (128, 63, 6, 0),
            64.0,
            (Vec3::new(-4.0, 0.0, -4.0), Vec3::new(4.0, 0.0, 4.0)),
            (352, 0, 0),
            ([0, 1, 10], 102),
            ([80, 79, 70], 102),
        ),
        (
            "waves",
            waves(),
            (512, 255, 8, 0),
            64.671_4,
            (Vec3::new(-4.0, -0.2999, -4.0), Vec3::new(4.0, 0.2999, 4.0)),
            (1400, 36, 36),
            ([0, 1, 18], 102),
            ([288, 287, 270], 102),
        ),
        (
            "valley",
            valley(),
            (4, 1, 1, 0),
            9.237_6,
            (Vec3::new(-2.0, 0.0, -1.0), Vec3::new(2.0, 1.1547, 1.0)),
            (4, 2, 0),
            ([0, 1, 2], 34),
            ([3, 5, 4], 68),
        ),
    ];
    for (what, mesh, counts, area, bounds, edges, first, last) in cases {
        assert_eq!(
            (
                mesh.triangle_count(),
                mesh.node_count(),
                mesh.tree_height(),
                mesh.degenerate_count()
            ),
            counts,
            "{what}: Box3D's triangle, node, height and degenerate counts"
        );
        assert!(
            (mesh.surface_area() - area).abs() <= 1e-3,
            "{what}: area {}",
            mesh.surface_area()
        );
        let (lower, upper) = mesh.bounds();
        assert!(
            (lower - bounds.0).abs().max_element() <= 1e-4
                && (upper - bounds.1).abs().max_element() <= 1e-4,
            "{what}: bounds {lower:?} {upper:?}"
        );
        assert_eq!(edge_counts(&mesh), edges, "{what}: flat, concave, convex");
        let n = mesh.triangle_count();
        for (index, (indices, flags)) in [(0, first), (n - 1, last)] {
            let t = mesh.triangle(index);
            assert_eq!(
                (t.indices, t.flags),
                (indices, flags),
                "{what}: triangle {index}"
            );
        }
    }

    // Box3D: a triangle with a repeated vertex is dropped without counting as degenerate.
    let vertices = [
        Vec3::ZERO,
        Vec3::Z,
        Vec3::X,
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(2.0, 0.0, 0.0),
    ];
    let strip = Mesh3d::new(&vertices, &[[0, 1, 2], [2, 2, 3], [2, 3, 4]]).unwrap();
    assert_eq!((strip.triangle_count(), strip.degenerate_count()), (2, 0));
}

#[test]
fn rays_hit_mesh_fronts() {
    let mesh = waves();
    // Box3D: (fraction, triangle, point, normal) for each ray, or a miss.
    let cases = [
        (
            "down",
            Vec3::new(0.3, 2.0, 0.7),
            Vec3::new(0.0, -4.0, 0.0),
            1.0,
            Some((
                0.484_778,
                387,
                Vec3::new(0.3, 0.060_89, 0.7),
                Vec3::new(-0.217_76, 0.975_55, 0.029_63),
            )),
        ),
        // Triangles face up: a ray from below passes.
        (
            "up",
            Vec3::new(0.3, -2.0, 0.7),
            Vec3::new(0.0, 4.0, 0.0),
            1.0,
            None,
        ),
        (
            "slant",
            Vec3::new(-3.0, 1.0, -2.0),
            Vec3::new(5.0, -1.5, 3.0),
            1.0,
            Some((
                0.637_478,
                326,
                Vec3::new(0.187_39, 0.043_78, -0.087_57),
                Vec3::new(-0.227_52, 0.973_77, 0.0),
            )),
        ),
        (
            "short",
            Vec3::new(0.3, 2.0, 0.7),
            Vec3::new(0.0, -4.0, 0.0),
            0.4,
            None,
        ),
    ];
    for (what, origin, translation, max_fraction, expected) in cases {
        let hit = mesh.ray_cast(origin, translation, max_fraction);
        match (hit, expected) {
            (None, None) => {}
            (Some(hit), Some((fraction, triangle, point, normal))) => {
                assert!(
                    (hit.fraction - fraction).abs() <= 1e-5,
                    "{what}: {}",
                    hit.fraction
                );
                assert_eq!(hit.triangle, triangle, "{what}: Box3D's triangle");
                assert!(
                    (hit.point - point).length() <= 1e-4,
                    "{what}: {:?}",
                    hit.point
                );
                assert!(
                    (hit.normal - normal).length() <= 1e-4,
                    "{what}: {:?}",
                    hit.normal
                );
            }
            (hit, _) => panic!("{what}: Box3D's hit is {expected:?}, Korppi's {hit:?}"),
        }
    }
}

fn world_with_mesh(mesh: Mesh3d, friction: f32, gravity: f32) -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, gravity, 0.0));
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        ground,
        Collider::new(Shape::TriangleMesh(mesh)).with_friction(friction),
    );
    world
}

fn dynamic(world: &mut PhysicsWorld, body: RigidBody, shape: Shape, friction: f32) -> u32 {
    let id = world.create_body(body.with_enable_sleep(false));
    world.create_collider(id, Collider::new(shape).with_friction(friction));
    id.into()
}

/// The manifold and point counts of a body's contacts.
fn contact_counts(world: &PhysicsWorld, id: u32) -> (usize, usize) {
    world
        .contacts()
        .filter(|c| c.body_a == id || c.body_b == id)
        .flat_map(|c| c.manifolds())
        .fold((0, 0), |(m, p), manifold| {
            (m + 1, p + manifold.points.len())
        })
}

const CUBE: Shape = Shape::Box {
    half_extents: Vec3::splat(0.5),
};

#[test]
fn boxes_slide_over_internal_edges() {
    // Box3D: a frictionless box sliding diagonally over a grid of 1 m cells crosses its edges
    // without a bump: |vy| stays below 0.00041 and |w| below 0.00021 after 10 steps, and after
    // 2 s it is at (3.99976, 0.50494, 4.99985) on 1 manifold of 4 points.
    let mut w = world_with_mesh(grid(40, 1.0), 0.0, -9.81);
    let mut body = RigidBody::new(BodyType::Dynamic, Vec3::new(-8.0, 0.505, -3.0));
    body.linear_velocity = Vec3::new(6.0, 0.0, 4.0);
    let id = dynamic(&mut w, body, CUBE, 0.0);
    let (mut max_vy, mut max_w) = (0.0f32, 0.0f32);
    for step in 0..120 {
        w.step(DT);
        let b = w.body(id).unwrap();
        if step >= 10 {
            max_vy = max_vy.max(b.linear_velocity.y.abs());
            max_w = max_w.max(b.angular_velocity.length());
        }
    }
    assert!(max_vy <= 0.002, "a ghost bump: |vy| reached {max_vy}");
    assert!(max_w <= 0.002, "a ghost bump: |w| reached {max_w}");
    let p = w.body(id).unwrap().position;
    assert!(
        (p - Vec3::new(3.999_76, 0.504_94, 4.999_85)).length() <= 1e-3,
        "{p:?}"
    );
    assert_eq!(contact_counts(&w, id), (1, 4));
}

#[test]
fn balls_roll_over_internal_edges() {
    // Box3D: a ball rolling over a grid of 1 m cells holds its height, 0.25089, and its
    // end-of-step velocity, from step 20 on; after 2 s it is at (2.00546, 0.25089, 4.00108).
    // The contact anchor turns with the ball during a step, so the last substeps let it approach
    // at 0.12 m/s and the next collide pushes it back, as in Box3D.
    let mut w = world_with_mesh(grid(40, 1.0), 0.6, -9.81);
    let r = 0.25;
    let mut body = RigidBody::new(BodyType::Dynamic, Vec3::new(-8.0, r + 0.005, 2.0));
    let v0 = Vec3::new(5.0, 0.0, 1.0);
    body.linear_velocity = v0;
    body.angular_velocity = Vec3::new(v0.z / r, 0.0, -v0.x / r);
    let id = dynamic(&mut w, body, Shape::Sphere { radius: r }, 0.6);
    let (mut heights, mut vys) = ((f32::MAX, f32::MIN), (f32::MAX, f32::MIN));
    for step in 0..120 {
        w.step(DT);
        let b = w.body(id).unwrap();
        if step >= 20 {
            heights = (heights.0.min(b.position.y), heights.1.max(b.position.y));
            vys = (
                vys.0.min(b.linear_velocity.y),
                vys.1.max(b.linear_velocity.y),
            );
        }
    }
    assert!(heights.1 - heights.0 <= 1e-4, "a ghost bump: {heights:?}");
    assert!(vys.1 - vys.0 <= 1e-3, "a ghost bump: {vys:?}");
    let p = w.body(id).unwrap().position;
    assert!(
        (p - Vec3::new(2.005_46, 0.250_89, 4.001_08)).length() <= 1e-3,
        "{p:?}"
    );
}

#[test]
fn shapes_rest_on_meshes() {
    // Box3D: a lying capsule, a cube and a ball dropped 0.2 m onto a grid rest one linear slop
    // above it, the mesh rest offset, at 0.10495, 0.50493 and 0.20493, on 1 manifold of 4, 4
    // and 1 points.
    let mut w = world_with_mesh(grid(10, 1.0), 0.6, -9.81);
    let capsule = dynamic(
        &mut w,
        RigidBody::new(BodyType::Dynamic, Vec3::new(-2.2, 0.3, 0.3))
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        Shape::Capsule {
            radius: 0.1,
            half_height: 0.3,
        },
        0.6,
    );
    let cube = dynamic(
        &mut w,
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.4, 0.7, 0.2)),
        CUBE,
        0.6,
    );
    let ball = dynamic(
        &mut w,
        RigidBody::new(BodyType::Dynamic, Vec3::new(2.5, 0.4, -1.5)),
        Shape::Sphere { radius: 0.2 },
        0.6,
    );
    for _ in 0..120 {
        w.step(DT);
    }
    for (what, id, height, points) in [
        ("capsule", capsule, 0.104_95, 4),
        ("cube", cube, 0.504_93, 4),
        ("ball", ball, 0.204_93, 1),
    ] {
        let y = w.body(id).unwrap().position.y;
        assert!((y - height).abs() <= 2e-4, "{what}: {y}");
        assert_eq!(contact_counts(&w, id), (1, points), "{what}");
    }
}

#[test]
fn mesh_cost_follows_nearby_triangles() {
    // A ball resting on a 300 x 300 grid (180,000 triangles) costs about what it costs on a
    // 3 x 3 grid: the BVH finds the few triangles near it. Box3D rests it at 0.20493 on both.
    let mut costs = [0.0; 2];
    for (cost, n) in costs.iter_mut().zip([3, 300]) {
        let mut w = world_with_mesh(grid(n, 1.0), 0.6, -9.81);
        let ball = dynamic(
            &mut w,
            RigidBody::new(BodyType::Dynamic, Vec3::new(0.25, 0.3, 0.25)),
            Shape::Sphere { radius: 0.2 },
            0.6,
        );
        for _ in 0..60 {
            w.step(DT);
        }
        let y = w.body(ball).unwrap().position.y;
        assert!((y - 0.204_93).abs() <= 2e-4, "{n} x {n}: {y}");
        let start = Instant::now();
        for _ in 0..600 {
            w.step(DT);
        }
        *cost = start.elapsed().as_secs_f64();
    }
    assert!(
        costs[1] <= 4.0 * costs[0] + 0.01,
        "the large grid cost {:.4} s, the small one {:.4} s",
        costs[1],
        costs[0]
    );
}

#[test]
fn triangles_are_one_sided() {
    // Box3D: without gravity, a ball rising through a grid at 4 m/s passes it, at y = 1.0 after
    // 0.5 s; one falling onto it stops at 0.205.
    let mut w = world_with_mesh(grid(4, 1.0), 0.6, 0.0);
    let mut rising = RigidBody::new(BodyType::Dynamic, Vec3::new(-0.6, -1.0, 0.3));
    rising.linear_velocity = Vec3::new(0.0, 4.0, 0.0);
    let rising = dynamic(&mut w, rising, Shape::Sphere { radius: 0.2 }, 0.6);
    let mut falling = RigidBody::new(BodyType::Dynamic, Vec3::new(0.6, 1.0, -0.3));
    falling.linear_velocity = Vec3::new(0.0, -4.0, 0.0);
    let falling = dynamic(&mut w, falling, Shape::Sphere { radius: 0.2 }, 0.6);
    for _ in 0..30 {
        w.step(DT);
    }
    let a = w.body(rising).unwrap();
    assert!(
        (a.position.y - 1.0).abs() <= 1e-3 && (a.linear_velocity.y - 4.0).abs() <= 1e-3,
        "the rising ball should pass, it is at {:?} at {:?}",
        a.position,
        a.linear_velocity
    );
    let b = w.body(falling).unwrap();
    assert!(
        (b.position.y - 0.205).abs() <= 1e-3 && b.linear_velocity.y.abs() <= 1e-3,
        "the falling ball should stop, it is at {:?} at {:?}",
        b.position,
        b.linear_velocity
    );
}

#[test]
fn creases_split_manifolds() {
    // Box3D: a 0.6 m box dropped into the valley rests on both slopes at y = 0.47889, its
    // contact holding a manifold of 4 points on each, with normals (0.5, 0.86603, 0) and
    // (-0.5, 0.86603, 0).
    let mut w = world_with_mesh(valley(), 0.6, -9.81);
    let id = dynamic(
        &mut w,
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.0, 0.0)),
        Shape::Box {
            half_extents: Vec3::splat(0.3),
        },
        0.6,
    );
    for _ in 0..120 {
        w.step(DT);
    }
    let b = w.body(id).unwrap();
    assert!(
        b.linear_velocity.length() <= 0.01,
        "{:?}",
        b.linear_velocity
    );
    assert!((b.position.y - 0.478_89).abs() <= 1e-3, "{:?}", b.position);
    let contacts: Vec<_> = w
        .contacts()
        .filter(|c| (c.body_a == id || c.body_b == id) && c.is_touching)
        .collect();
    assert_eq!(contacts.len(), 1, "one contact with the mesh");
    let normals: Vec<(Vec3, usize)> = contacts[0]
        .manifolds()
        .iter()
        .map(|m| {
            // From the mesh to the box.
            let n = if contacts[0].body_b == id {
                m.normal
            } else {
                -m.normal
            };
            (n, m.points.len())
        })
        .collect();
    assert_eq!(normals.len(), 2, "one manifold per slope: {normals:?}");
    for ((normal, points), expected) in normals.iter().zip([
        Vec3::new(0.5, 0.866_03, 0.0),
        Vec3::new(-0.5, 0.866_03, 0.0),
    ]) {
        assert!((*normal - expected).length() <= 1e-3, "{normal:?}");
        assert_eq!(*points, 4);
    }
}
