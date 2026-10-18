//! The dynamic tree builds the same hierarchy as Box2D v3.2's (P2D-15).
//!
//! Before FIX-2.5 the tree was Box2D v2.4's: a sibling chosen by descending on the area cost,
//! then AVL-style height rotations, and every awake proxy reinserted each step. Box2D v3.2 picks
//! the sibling by branch and bound (`b2FindBestSibling`), rotates nodes to reduce the surface area
//! (`b2RotateNodes`), keeps moved flags in the nodes, grows and refits moved proxies in place,
//! and rebuilds the moved subtrees by median split every step (`b2DynamicTree_Rebuild`). The
//! tree's height, area ratio (internal node perimeters over the root's) and leaf visit order
//! show whether it builds the same hierarchy. Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_5_dynamic_tree.c`.

use korppi_math::{Aabb2d, Ray2d, Rot2, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::bvh::DynamicTree2d;
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d, WorldConfig2d};

/// The probe's linear congruential generator: 24 bits over 2^24, exact in C and Rust.
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / 16_777_216.0
    }

    fn random_box(&mut self) -> Aabb2d {
        let cx = 100.0 * self.next();
        let cy = 100.0 * self.next();
        let hx = 0.1 + self.next();
        let hy = 0.1 + self.next();
        Aabb2d::new(Vec2::new(cx - hx, cy - hy), Vec2::new(cx + hx, cy + hy))
    }
}

/// Asserts the proxy count, height and area ratio Box2D reports after `phase`. The area ratio
/// is Box2D's as printed, to six decimals.
fn assert_tree(tree: &DynamicTree2d, phase: &str, proxies: usize, height: i32, area_ratio: f64) {
    assert_eq!(
        (tree.proxy_count(), tree.height()),
        (proxies, height),
        "{phase}: (proxies, height)"
    );
    let ratio = f64::from(tree.area_ratio());
    assert!(
        (ratio - area_ratio).abs() < 1e-5,
        "{phase}: area ratio {ratio}, Box2D {area_ratio}"
    );
}

/// Leaves in visit order: how many, the sum of (visit number) × (user data), and the first six.
#[derive(Debug, Default, PartialEq)]
struct Visits {
    count: u64,
    checksum: u64,
    first: Vec<u32>,
}

impl Visits {
    fn visit(&mut self, user_data: u32) {
        if self.first.len() < 6 {
            self.first.push(user_data);
        }
        self.count += 1;
        self.checksum += self.count * u64::from(user_data);
    }

    fn of_query(tree: &DynamicTree2d) -> Self {
        let mut visits = Self::default();
        let query = Aabb2d::new(Vec2::new(30.0, 30.0), Vec2::new(50.0, 50.0));
        tree.query(&query, |_proxy, user_data| {
            visits.visit(user_data);
            true
        });
        visits
    }

    fn of_ray(tree: &DynamicTree2d) -> Self {
        let mut visits = Self::default();
        // Box2D's ray from (0, 50) along (100, 0); a negative return keeps it whole.
        let ray = Ray2d::new(Vec2::new(0.0, 50.0), Vec2::new(1.0, 0.0));
        tree.ray_cast(&ray, 100.0, |_proxy, user_data, _| {
            visits.visit(user_data);
            -1.0
        });
        visits
    }
}

fn visits(count: u64, checksum: u64, first: [u32; 6]) -> Visits {
    Visits {
        count,
        checksum,
        first: first.to_vec(),
    }
}

#[test]
fn the_tree_matches_box2d_through_every_operation() {
    let mut tree = DynamicTree2d::new();
    let mut rng = Lcg(12_345);
    let mut ids: Vec<Option<u32>> = Vec::new();

    // Insertions, with rotations.
    for i in 0..500 {
        ids.push(Some(tree.insert_proxy(rng.random_box(), i, false)));
    }
    assert_tree(&tree, "insert", 500, 14, 39.196_19);

    // Every third proxy moves: removed and inserted again, without rotations.
    for i in (0..500).step_by(3) {
        tree.reinsert_proxy(ids[i].unwrap(), rng.random_box(), false);
    }
    assert_tree(&tree, "move", 500, 15, 40.108_673);

    // Every fifth proxy from the second is destroyed.
    for i in (1..500).step_by(5) {
        tree.destroy_proxy(ids[i].take().unwrap());
    }
    assert_tree(&tree, "destroy", 400, 14, 35.238_541);

    // A hundred more reuse the freed proxy ids, last freed first.
    for i in 500..600 {
        ids.push(Some(tree.insert_proxy(rng.random_box(), i, false)));
    }
    assert_tree(&tree, "refill", 500, 13, 39.642_044);
    assert_eq!(&ids[500..503], &[Some(496), Some(491), Some(486)]);

    // Queries visit the leaves in Box2D's order; ray casts visit the nearer child first.
    assert_eq!(
        Visits::of_query(&tree),
        visits(17, 36_210, [494, 335, 165, 164, 591, 360])
    );
    assert_eq!(
        Visits::of_ray(&tree),
        visits(6, 5_651, [402, 388, 170, 130, 529, 133])
    );

    // A full rebuild sorts every leaf by median split.
    assert_eq!(tree.rebuild(true), 500);
    assert_tree(&tree, "full", 500, 12, 38.092_728);

    // Every seventh proxy grows; a partial rebuild sorts only the moved subtrees again.
    for &id in ids.iter().step_by(7).flatten() {
        let mut aabb = tree.proxy_aabb(id);
        aabb.min -= Vec2::new(0.5, 0.25);
        aabb.max += Vec2::new(0.25, 0.5);
        tree.enlarge_proxy(id, aabb);
    }
    assert_tree(&tree, "enlarge", 500, 12, 38.639_381);
    assert_eq!(tree.rebuild(false), 228);
    assert_tree(&tree, "partial", 500, 12, 39.727_833);

    // Every eleventh proxy jumps: its leaf takes the new box, its ancestors are marked, and the
    // refit grows them; then a partial rebuild.
    for &id in ids.iter().step_by(11).flatten() {
        let mut aabb = tree.proxy_aabb(id);
        aabb.min += Vec2::new(3.0, -2.0);
        aabb.max += Vec2::new(3.0, -2.0);
        tree.mark_proxy_moved(id, aabb);
    }
    tree.refit();
    assert_tree(&tree, "refit", 500, 12, 40.416_359);
    let root = tree.root_bounds();
    for (value, box2d) in [
        (root.min.x, -0.9805),
        (root.min.y, -1.5958),
        (root.max.x, 100.9246),
        (root.max.y, 100.7234),
    ] {
        assert!((value - box2d).abs() < 1e-4, "refit root {root:?}");
    }
    assert_eq!(tree.rebuild(false), 178);
    assert_tree(&tree, "partial2", 500, 13, 40.388_802);

    // Fifty proxies created moved, as the world creates them; the rebuild sorts them in.
    for i in 0..50 {
        tree.insert_proxy(rng.random_box(), 600 + i, true);
    }
    assert_tree(&tree, "moved", 550, 13, 41.943_974);
    assert_eq!(tree.rebuild(false), 194);
    assert_tree(&tree, "partial3", 550, 13, 43.254_81);
    assert_eq!(
        Visits::of_query(&tree),
        visits(20, 66_577, [187, 130, 165, 608, 35, 494])
    );
    assert_eq!(
        Visits::of_ray(&tree),
        visits(8, 12_157, [170, 402, 388, 130, 187, 529])
    );
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
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(hx, hy))),
    );
    id
}

/// Box2D v3.2's counters: (bodies, contacts, tree height, static tree height).
fn assert_counters(world: &PhysicsWorld2d, phase: &str, expected: (usize, usize, i32, i32)) {
    let c = world.counters();
    assert_eq!(
        (
            c.body_count,
            c.contact_count,
            c.tree_height,
            c.static_tree_height
        ),
        expected,
        "{phase}: (bodies, contacts, tree height, static tree height)"
    );
}

#[test]
fn the_world_trees_match_box2d() {
    // A hundred static boxes, 150 dynamic and five kinematic ones, none close enough to pair, in
    // a world without gravity: nothing moves unless the test moves it. The static tree grows by
    // insertion; the dynamic tree is rebuilt where proxies moved, at every step.
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::ZERO)
        .with_config(WorldConfig2d {
            enable_sleep: false,
            ..WorldConfig2d::default()
        });
    let mut rng = Lcg(777);
    for i in 0..10 {
        for j in 0..10 {
            let hx = 0.2 + 0.8 * rng.next();
            let hy = 0.2 + 0.8 * rng.next();
            add_box(
                &mut world,
                BodyType::Static,
                10.0 * i as f32,
                10.0 * j as f32,
                hx,
                hy,
            );
        }
    }
    let mut dynamic = Vec::new();
    for i in 0..10 {
        for j in 0..15 {
            let hx = 0.2 + 0.8 * rng.next();
            let hy = 0.2 + 0.8 * rng.next();
            let (x, y) = (10.0 * i as f32 + 5.0, 6.0 * j as f32 + 3.0);
            dynamic.push(add_box(&mut world, BodyType::Dynamic, x, y, hx, hy));
        }
    }
    for i in 0..5 {
        add_box(
            &mut world,
            BodyType::Kinematic,
            12.0 * i as f32,
            -20.0,
            1.0,
            1.0,
        );
    }
    let dt = 1.0 / 60.0;
    assert_counters(&world, "created", (255, 0, 10, 9));
    world.step(dt);
    assert_counters(&world, "step 1", (255, 0, 8, 9));

    // Every fourth dynamic body jumps to an empty region: removed and inserted again.
    for k in (0..150).step_by(4) {
        let position = Vec2::new(130.0 + 4.0 * (k % 20) as f32, 4.0 * (k / 20) as f32);
        world.set_transform(dynamic[k], position, Rot2::IDENTITY);
    }
    assert_counters(&world, "teleported", (255, 0, 11, 9));
    world.step(dt);
    assert_counters(&world, "step 2", (255, 0, 9, 9));

    // Every fifth from the second is destroyed.
    for k in (1..150).step_by(5) {
        world.destroy_body(dynamic[k]);
    }
    world.step(dt);
    assert_counters(&world, "step 3", (225, 0, 9, 9));

    // Thirty more in a row below.
    for k in 0..30 {
        add_box(
            &mut world,
            BodyType::Dynamic,
            4.0 * k as f32,
            -10.0,
            0.5,
            0.5,
        );
    }
    assert_counters(&world, "added", (255, 0, 10, 9));
    world.step(dt);
    assert_counters(&world, "step 4", (255, 0, 11, 9));
}
