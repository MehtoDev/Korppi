//! The acoustic scene from the physics world's static colliders (FIX-7.5, ADR-051; finding
//! AUD-08).
//!
//! The owner decided that raytraced mode traces the level's physics colliders. The acoustics
//! worker cannot query the physics world while it steps, so an [`AcousticGeometry`] is a
//! snapshot of the static colliders (shape, world pose and acoustic material) in the physics
//! crate's bounding volume hierarchy, ray cast through the shapes' own ray casts. It is built on
//! the game thread when the static colliders change and shared through an `Arc`.
//!
//! Inputs: a [`PhysicsWorld`] and each collider's material. Outputs: closest hits, transmission
//! through what a segment crosses, precomputed diffraction edges ([`Wedge`]), and occlusion
//! ([`AcousticRaycaster`]).

use super::diffraction::Wedge;
use super::geometry::{AcousticHit, AcousticRay};
use super::material::AcousticMaterial;
use crate::spatial::AcousticRaycaster;
use korppi_math::{Quat, Ray, Vec3};
use korppi_physics::{BodyType, ColliderId3d, DynamicTree, Mesh3d, MeshEdge, PhysicsWorld, Shape};

/// How close a crossing may be to a segment's ends and still not count.
const EPSILON: f32 = 1e-4;
/// The most triangle layers of one mesh a segment is counted through.
const MAX_LAYERS: usize = 32;

/// A static collider as the acoustics see it.
#[derive(Debug, Clone)]
struct Surface {
    shape: Shape,
    position: Vec3,
    rotation: Quat,
    material: AcousticMaterial,
    wedges: Vec<Wedge>,
    wedge_pairs: Vec<(usize, usize)>,
}

fn box_edges(half_extents: Vec3) -> (Vec<Wedge>, Vec<(usize, usize)>) {
    let (hx, hy, hz) = (half_extents.x, half_extents.y, half_extents.z);
    let angle = 1.5 * std::f32::consts::PI;
    let mut wedges = Vec::with_capacity(12);

    for &sy in &[1.0, -1.0] {
        for &sz in &[1.0, -1.0] {
            wedges.push(Wedge {
                start: Vec3::new(-hx, sy * hy, sz * hz),
                end: Vec3::new(hx, sy * hy, sz * hz),
                face: -sz * Vec3::Z,
                normal: sy * Vec3::Y,
                angle,
            });
        }
    }

    for &sx in &[1.0, -1.0] {
        for &sz in &[1.0, -1.0] {
            wedges.push(Wedge {
                start: Vec3::new(sx * hx, -hy, sz * hz),
                end: Vec3::new(sx * hx, hy, sz * hz),
                face: -sz * Vec3::Z,
                normal: sx * Vec3::X,
                angle,
            });
        }
    }

    for &sx in &[1.0, -1.0] {
        for &sy in &[1.0, -1.0] {
            wedges.push(Wedge {
                start: Vec3::new(sx * hx, sy * hy, -hz),
                end: Vec3::new(sx * hx, sy * hy, hz),
                face: -sy * Vec3::Y,
                normal: sx * Vec3::X,
                angle,
            });
        }
    }

    let pairs = vec![
        (0, 1),
        (8, 10),
        (2, 3),
        (9, 11),
        (4, 5),
        (8, 9),
        (6, 7),
        (10, 11),
        (0, 2),
        (4, 6),
        (1, 3),
        (5, 7),
    ];
    (wedges, pairs)
}

fn mesh_edges(mesh: &Mesh3d) -> (Vec<Wedge>, Vec<(usize, usize)>) {
    let mut wedges = Vec::new();
    let mut convex_map: std::collections::HashMap<(u32, u32), (Vec3, Vec3, Vec3)> =
        std::collections::HashMap::new();

    for t in mesh.triangles() {
        let normal = (t.vertices[1] - t.vertices[0])
            .cross(t.vertices[2] - t.vertices[0])
            .normalize_or_zero();
        for e in 0..3 {
            let (v0, v1) = (t.vertices[e], t.vertices[(e + 1) % 3]);
            let edge_vec = v1 - v0;
            if edge_vec.length_squared() < 1e-8 {
                continue;
            }
            let (i0, i1) = (t.indices[e], t.indices[(e + 1) % 3]);
            let key = (i0.min(i1), i0.max(i1));
            match t.edge(e) {
                MeshEdge::Open => {
                    let u = edge_vec.normalize_or_zero();
                    let face = normal.cross(u).normalize_or_zero();
                    wedges.push(Wedge {
                        start: v0,
                        end: v1,
                        face,
                        normal,
                        angle: 2.0 * std::f32::consts::PI,
                    });
                }
                MeshEdge::Convex => {
                    if let Some((prev_v0, prev_v1, prev_norm)) = convex_map.remove(&key) {
                        let u = (prev_v1 - prev_v0).normalize_or_zero();
                        let face = prev_norm.cross(u).normalize_or_zero();
                        let phi = prev_norm.dot(normal).clamp(-1.0, 1.0).acos();
                        wedges.push(Wedge {
                            start: prev_v0,
                            end: prev_v1,
                            face,
                            normal: prev_norm,
                            angle: std::f32::consts::PI + phi,
                        });
                    } else {
                        convex_map.insert(key, (v0, v1, normal));
                    }
                }
                _ => {}
            }
        }
    }
    (wedges, Vec::new())
}

fn shape_edges(shape: &Shape, position: Vec3, rotation: Quat) -> (Vec<Wedge>, Vec<(usize, usize)>) {
    let (local_wedges, pairs) = match shape {
        Shape::Box { half_extents } => box_edges(*half_extents),
        Shape::TriangleMesh(mesh) => mesh_edges(mesh),
        Shape::Triangle { a, b, c } => {
            let normal = (*b - *a).cross(*c - *a).normalize_or_zero();
            let verts = [*a, *b, *c];
            let mut wedges = Vec::with_capacity(3);
            for i in 0..3 {
                let (v0, v1) = (verts[i], verts[(i + 1) % 3]);
                let u = (v1 - v0).normalize_or_zero();
                let face = normal.cross(u).normalize_or_zero();
                wedges.push(Wedge {
                    start: v0,
                    end: v1,
                    face,
                    normal,
                    angle: 2.0 * std::f32::consts::PI,
                });
            }
            (wedges, Vec::new())
        }
        _ => (Vec::new(), Vec::new()),
    };

    let world_wedges = local_wedges
        .into_iter()
        .map(|w| Wedge {
            start: position + rotation * w.start,
            end: position + rotation * w.end,
            face: (rotation * w.face).normalize_or_zero(),
            normal: (rotation * w.normal).normalize_or_zero(),
            angle: w.angle,
        })
        .collect();

    (world_wedges, pairs)
}

impl Surface {
    /// The hit of `ray` within `max_distance`: its distance and the normal facing the ray.
    fn hit(&self, ray: &AcousticRay, max_distance: f32) -> Option<(f32, Vec3)> {
        let inverse = self.rotation.conjugate();
        let local = Ray::new(
            inverse * (ray.origin - self.position),
            inverse * ray.direction,
        );
        let (distance, normal, _) = self.shape.ray_cast_detailed(&local, max_distance)?;
        let mut normal = (self.rotation * normal).normalize_or_zero();
        if normal.dot(ray.direction) > 0.0 {
            normal = -normal;
        }
        Some((distance, normal))
    }

    /// Whether the shape is a sheet of triangles, crossed once per layer, rather than a solid,
    /// crossed once.
    fn is_layered(&self) -> bool {
        matches!(self.shape, Shape::TriangleMesh(_) | Shape::HeightField(_))
    }

    /// How many times the segment of `ray` up to `length` crosses the surface.
    fn crossings(&self, ray: &AcousticRay, length: f32) -> usize {
        if !self.is_layered() {
            return usize::from(
                self.hit(ray, length - EPSILON)
                    .is_some_and(|(distance, _)| distance > EPSILON),
            );
        }
        let (mut start, mut layers) = (0.0, 0);
        while layers < MAX_LAYERS {
            let from = AcousticRay {
                origin: ray.at(start),
                direction: ray.direction,
            };
            let Some((distance, _)) = self.hit(&from, length - start) else {
                break;
            };
            if start + distance >= length - EPSILON {
                break;
            }
            if start + distance > EPSILON {
                layers += 1;
            }
            start += distance + EPSILON;
        }
        layers
    }
}

/// The static colliders of a physics world, for the acoustics (see the module documentation).
/// Shared through an `Arc`, not cloned.
#[derive(Debug)]
pub struct AcousticGeometry {
    surfaces: Vec<Surface>,
    tree: DynamicTree,
}

impl AcousticGeometry {
    /// A snapshot of `world`'s static colliders (sensors excluded), each with the material
    /// `material` gives its collider, or the default material.
    #[must_use]
    pub fn from_physics(
        world: &PhysicsWorld,
        mut material: impl FnMut(ColliderId3d) -> Option<AcousticMaterial>,
    ) -> Self {
        let mut surfaces = Vec::new();
        let mut tree = DynamicTree::new();
        for (id, collider) in world.colliders() {
            let Some(body) = world.collider_body(id).and_then(|body| world.body(body)) else {
                continue;
            };
            if body.body_type != BodyType::Static || collider.is_sensor {
                continue;
            }
            let (position, rotation) = collider.world_pose(body.position, body.rotation);
            tree.create_proxy(
                collider.shape.compute_aabb(position, rotation),
                surfaces.len() as u32,
            );
            let (wedges, wedge_pairs) = shape_edges(&collider.shape, position, rotation);
            surfaces.push(Surface {
                shape: collider.shape.clone(),
                position,
                rotation,
                material: material(id).unwrap_or_default(),
                wedges,
                wedge_pairs,
            });
        }
        Self { surfaces, tree }
    }

    /// The number of static colliders in the snapshot.
    #[must_use]
    pub fn surface_count(&self) -> usize {
        self.surfaces.len()
    }

    /// The precomputed diffraction edges of surface `index` in world space.
    #[must_use]
    pub fn surface_wedges(&self, index: usize) -> &[Wedge] {
        self.surfaces
            .get(index)
            .map_or(&[], |surface| &surface.wedges)
    }

    /// The face-sharing edge index pairs of surface `index`, for second-order diffraction.
    #[must_use]
    pub fn surface_wedge_pairs(&self, index: usize) -> &[(usize, usize)] {
        self.surfaces
            .get(index)
            .map_or(&[], |surface| &surface.wedge_pairs)
    }

    /// The indices of every surface crossed by the segment from `from` to `to`.
    #[must_use]
    pub fn crossed_surfaces(&self, from: Vec3, to: Vec3) -> Vec<usize> {
        let delta = to - from;
        let distance = delta.length();
        if distance < EPSILON {
            return Vec::new();
        }
        let ray = AcousticRay::new(from, delta / distance);
        let mut crossed = Vec::new();
        self.tree.ray_cast(
            &Ray::new(ray.origin, ray.direction),
            distance,
            |index, _| {
                let surface = &self.surfaces[index as usize];
                if surface.crossings(&ray, distance) > 0 {
                    crossed.push(index as usize);
                }
                distance
            },
        );
        crossed
    }

    /// The closest surface `ray` hits within `max_distance`.
    #[must_use]
    pub fn raycast_closest(&self, ray: &AcousticRay, max_distance: f32) -> Option<AcousticHit> {
        let mut closest = None;
        let mut max = max_distance;
        self.tree.ray_cast(
            &Ray::new(ray.origin, ray.direction),
            max_distance,
            |index, _| {
                let surface = &self.surfaces[index as usize];
                if let Some((distance, normal)) = surface.hit(ray, max) {
                    if distance < max {
                        max = distance;
                        closest = Some(AcousticHit {
                            distance,
                            point: ray.at(distance),
                            normal,
                            material: surface.material,
                        });
                    }
                }
                max
            },
        );
        closest
    }

    /// The product, per band, of the transmission of every surface the segment of `ray` up to
    /// `max_distance` crosses: once per solid, once per triangle layer of a mesh.
    #[must_use]
    pub fn raycast_transmission(&self, ray: &AcousticRay, max_distance: f32) -> [f32; 3] {
        let mut transmission = [1.0f32; 3];
        self.tree.ray_cast(
            &Ray::new(ray.origin, ray.direction),
            max_distance,
            |index, _| {
                let surface = &self.surfaces[index as usize];
                for _ in 0..surface.crossings(ray, max_distance) {
                    for (band, through) in transmission.iter_mut().enumerate() {
                        *through *= surface.material.transmission[band];
                    }
                }
                max_distance
            },
        );
        transmission
    }
}

impl AcousticRaycaster for AcousticGeometry {
    /// How much of the line from `from` to `to` is blocked: 1 − the mean transmission of what it
    /// crosses, or `None` if nothing blocks it.
    fn raycast_occlusion(&self, from: Vec3, to: Vec3) -> Option<f32> {
        let delta = to - from;
        let distance = delta.length();
        if distance < EPSILON {
            return None;
        }
        let ray = AcousticRay::new(from, delta / distance);
        self.raycast_closest(&ray, distance - EPSILON)?;
        let through = self.raycast_transmission(&ray, distance);
        let occlusion = (1.0 - (through[0] + through[1] + through[2]) / 3.0).clamp(0.0, 1.0);
        (occlusion > EPSILON).then_some(occlusion)
    }
}
