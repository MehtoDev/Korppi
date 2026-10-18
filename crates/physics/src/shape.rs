//! Geometric collision shapes.

use crate::compound::Compound3d;
use crate::height_field::HeightField3d;
use crate::hull::Hull3d;
use crate::mesh::Mesh3d;
use korppi_math::{Aabb, Mat3, Quat, Ray, Vec3};
use serde::{Deserialize, Serialize};

/// Geometric shape used for collision detection and mass computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Shape {
    /// A 3D sphere centered at local origin.
    Sphere {
        /// Radius of the sphere.
        radius: f32,
    },
    /// A 3D box (cuboid) centered at local origin.
    Box {
        /// Half extents along each local axis (x, y, z).
        half_extents: Vec3,
    },
    /// A 3D capsule aligned along the local Y axis.
    Capsule {
        /// Radius of the hemispherical caps.
        radius: f32,
        /// Half distance between sphere centers along the Y axis.
        half_height: f32,
    },
    /// A convex polyhedron with Box3D's hull topology, built from points by
    /// [`Shape::convex_hull`] or [`Hull3d::new`].
    ConvexHull(Hull3d),
    /// A static triangle mesh with Box3D's BVH and edge flags, built by
    /// [`Shape::triangle_mesh`] or [`Mesh3d::new`]. Its triangles are one-sided, facing where
    /// their vertices turn counter-clockwise; it collides with spheres, capsules, boxes and
    /// hulls, not with other meshes or triangles, as in Box3D.
    TriangleMesh(Mesh3d),
    /// A static terrain height field with Box3D's grid topology and edge flags, built by
    /// [`HeightField3d::new`].
    HeightField(HeightField3d),
    /// A static aggregate shape containing multiple child shapes with an internal BVH, built
    /// by [`Compound3d::new`].
    Compound(Compound3d),
    /// A single one-sided triangle in local coordinates, colliding as a one-triangle mesh.
    Triangle {
        /// First vertex.
        a: Vec3,
        /// Second vertex.
        b: Vec3,
        /// Third vertex.
        c: Vec3,
    },
}

impl Shape {
    /// The convex hull of `points` (Box3D `b3CreateHull`), or `None` when they span no volume.
    #[must_use]
    pub fn convex_hull(points: &[Vec3]) -> Option<Self> {
        Hull3d::new(points).map(Self::ConvexHull)
    }

    /// The mesh of `vertices` and counter-clockwise triangles `indices` (Box3D `b3CreateMesh`
    /// with identified edges), or `None` when no triangle has area or an index is out of range.
    #[must_use]
    pub fn triangle_mesh(vertices: &[Vec3], indices: &[[u32; 3]]) -> Option<Self> {
        Mesh3d::new(vertices, indices).map(Self::TriangleMesh)
    }

    /// Whether the shape is convex: every shape but a triangle mesh, height field or compound (Box3D `b3IsConvex`).
    #[must_use]
    pub const fn is_convex(&self) -> bool {
        !matches!(
            self,
            Self::TriangleMesh(_) | Self::HeightField(_) | Self::Compound(_)
        )
    }

    /// Whether the shape is made of one-sided triangles: a mesh, a height field, or a triangle.
    #[must_use]
    pub const fn is_triangles(&self) -> bool {
        matches!(
            self,
            Self::TriangleMesh(_) | Self::HeightField(_) | Self::Triangle { .. }
        )
    }

    /// The radius that scales rolling resistance (Box3D's contact rolling radius): a sphere's
    /// or capsule's radius and a quarter of a hull's inner radius (a box's smallest half
    /// extent). Triangles, meshes, and compounds have none.
    #[must_use]
    pub fn rolling_radius(&self) -> f32 {
        match self {
            Self::Sphere { radius } | Self::Capsule { radius, .. } => *radius,
            Self::Box { half_extents } => 0.25 * half_extents.min_element(),
            Self::ConvexHull(hull) => 0.25 * hull.inner_radius(),
            _ => 0.0,
        }
    }

    /// The shape's extents about its origin (Box3D `b3ComputeShapeExtent`): the smallest, the
    /// radius of a sphere inside the shape (a hull's inner radius, a box's smallest half
    /// extent), and the largest per axis. A triangle's smallest extent is the linear slop; a
    /// triangle mesh is never swept, so its smallest extent is unbounded.
    #[must_use]
    pub fn extents(&self) -> (f32, Vec3) {
        let max_abs = |points: &[Vec3]| {
            points
                .iter()
                .fold(Vec3::ZERO, |extent, p| extent.max(p.abs()))
        };
        match self {
            Self::Sphere { radius } => (*radius, Vec3::splat(*radius)),
            Self::Capsule {
                radius,
                half_height,
            } => (*radius, Vec3::new(*radius, half_height + radius, *radius)),
            Self::Box { half_extents } => (half_extents.min_element(), *half_extents),
            Self::ConvexHull(hull) => (hull.inner_radius(), max_abs(hull.points())),
            Self::Triangle { a, b, c } => {
                (crate::contact_solver::LINEAR_SLOP, max_abs(&[*a, *b, *c]))
            }
            Self::TriangleMesh(mesh) => (f32::MAX, max_abs(mesh.vertices())),
            Self::HeightField(hf) => (f32::MAX, hf.bounds().1),
            Self::Compound(comp) => (f32::MAX, comp.bounds().1),
        }
    }

    /// Broadphase fat-AABB margin for this shape on a non-static body (Box3D `b3ComputeShapeMargin`).
    #[must_use]
    pub fn aabb_margin(&self) -> f32 {
        const MAX_AABB_MARGIN: f32 = 0.05;
        const AABB_MARGIN_FRACTION: f32 = 0.125;
        let extent = match self {
            Self::Sphere { radius } => *radius,
            Self::Capsule {
                radius,
                half_height,
            } => *half_height + *radius,
            Self::Box { half_extents } => half_extents.length(),
            Self::ConvexHull(hull) => {
                let c = hull.center();
                hull.points()
                    .iter()
                    .map(|&p| (p - c).length_squared())
                    .fold(0.0f32, f32::max)
                    .sqrt()
            }
            Self::Triangle { a, b, c } => {
                let center = (*a + *b + *c) * (1.0 / 3.0);
                [*a, *b, *c]
                    .iter()
                    .map(|&p| (p - center).length_squared())
                    .fold(0.0f32, f32::max)
                    .sqrt()
            }
            Self::TriangleMesh(_) | Self::HeightField(_) | Self::Compound(_) => {
                return MAX_AABB_MARGIN;
            }
        };
        MAX_AABB_MARGIN.min(AABB_MARGIN_FRACTION * extent)
    }

    /// Projected surface area of this shape onto a plane with normal `plane_normal` in the
    /// shape's frame (Box3D `b3GetShapeProjectedArea`).
    #[must_use]
    pub fn projected_area(&self, plane_normal: Vec3) -> f32 {
        use std::f32::consts::PI;
        match self {
            Self::Sphere { radius } => PI * radius * radius,
            Self::Capsule {
                radius,
                half_height,
            } => {
                let axis = Vec3::new(0.0, 2.0 * half_height, 0.0);
                let projected_length = axis.cross(plane_normal).length();
                let cylinder_area = 2.0 * radius * projected_length;
                let sphere_area = PI * radius * radius;
                sphere_area + cylinder_area
            }
            Self::Box { half_extents } => {
                let h = half_extents;
                let n = plane_normal;
                2.0 * (h.y * h.z * n.x.abs() + h.x * h.z * n.y.abs() + h.x * h.y * n.z.abs())
            }
            Self::ConvexHull(hull) => 0.25 * hull.surface_area(),
            _ => 0.0,
        }
    }

    /// Computes the world-space bounding box for this shape given a position and orientation.
    #[must_use]
    pub fn compute_aabb(&self, position: Vec3, rotation: Quat) -> Aabb {
        match self {
            Self::Sphere { radius } => Aabb {
                min: position - Vec3::splat(*radius),
                max: position + Vec3::splat(*radius),
            },
            Self::Box { half_extents } => {
                let r_mat = korppi_math::Mat3::from_quat(rotation);
                let x_axis = (r_mat.x_axis * half_extents.x).abs();
                let y_axis = (r_mat.y_axis * half_extents.y).abs();
                let z_axis = (r_mat.z_axis * half_extents.z).abs();
                let world_half_extents = x_axis + y_axis + z_axis;

                Aabb {
                    min: position - world_half_extents,
                    max: position + world_half_extents,
                }
            }
            Self::Capsule {
                radius,
                half_height,
            } => {
                let y_axis = rotation * Vec3::new(0.0, *half_height, 0.0);
                let p1 = position - y_axis;
                let p2 = position + y_axis;
                let min = p1.min(p2) - Vec3::splat(*radius);
                let max = p1.max(p2) + Vec3::splat(*radius);
                Aabb { min, max }
            }
            Self::ConvexHull(hull) => transform_bounds(hull.bounds(), position, rotation),
            // b3ComputeMeshAABB.
            Self::TriangleMesh(mesh) => transform_bounds(mesh.bounds(), position, rotation),
            Self::HeightField(hf) => transform_bounds(hf.bounds(), position, rotation),
            Self::Compound(comp) => transform_bounds(comp.bounds(), position, rotation),
            Self::Triangle { a, b, c } => {
                let w_a = position + rotation * *a;
                let w_b = position + rotation * *b;
                let w_c = position + rotation * *c;
                Aabb {
                    min: w_a.min(w_b).min(w_c),
                    max: w_a.max(w_b).max(w_c),
                }
            }
        }
    }

    /// The shape's mass, centre of mass and inertia tensor about it, in the shape's frame, at
    /// `density` (Box3D `b3ComputeSphereMass`, `b3ComputeCapsuleMass`, `b3ComputeHullMass` of a
    /// box hull); triangles and meshes have none, as Box3D's meshes.
    #[must_use]
    pub fn mass_data(&self, density: f32) -> (f32, Vec3, Mat3) {
        use std::f32::consts::PI;
        match self {
            Self::Sphere { radius } => {
                let r = *radius;
                let mass = 4.0 / 3.0 * PI * r * r * r * density;
                let i = 0.4 * mass * r * r;
                (mass, Vec3::ZERO, Mat3::from_diagonal(Vec3::splat(i)))
            }
            Self::Capsule {
                radius,
                half_height,
            } => {
                let r = *radius;
                let height = 2.0 * half_height;
                let cylinder_mass = PI * r * r * height * density;
                let sphere_mass = 4.0 / 3.0 * PI * r * r * r * density;
                // b3CylinderInertia + b3SphereInertia, the hemispheres moved to the ends.
                let cylinder_across = cylinder_mass * (3.0 * r * r + height * height) / 12.0;
                let cylinder_along = 0.5 * cylinder_mass * r * r;
                let sphere = 0.4 * sphere_mass * r * r;
                let steiner = 0.125 * sphere_mass * (3.0 * r + 2.0 * height) * height;
                let across = cylinder_across + sphere + steiner;
                (
                    sphere_mass + cylinder_mass,
                    Vec3::ZERO,
                    Mat3::from_diagonal(Vec3::new(across, cylinder_along + sphere, across)),
                )
            }
            Self::Box { half_extents } => {
                let h = half_extents.max(Vec3::splat(0.2 * crate::contact_solver::LINEAR_SLOP));
                let volume = 8.0 * h.x * h.y * h.z;
                let d = 2.0 * h;
                // b3BoxInertia(volume, -h, h) times the density.
                let central = Vec3::new(
                    volume * (d.y * d.y + d.z * d.z) / 12.0,
                    volume * (d.x * d.x + d.z * d.z) / 12.0,
                    volume * (d.x * d.x + d.y * d.y) / 12.0,
                );
                (
                    density * volume,
                    Vec3::ZERO,
                    Mat3::from_diagonal(central * density),
                )
            }
            // b3ComputeHullMass.
            Self::ConvexHull(hull) => (
                density * hull.volume(),
                hull.center(),
                hull.central_inertia() * density,
            ),
            Self::TriangleMesh(_)
            | Self::HeightField(_)
            | Self::Compound(_)
            | Self::Triangle { .. } => (0.0, Vec3::ZERO, Mat3::ZERO),
        }
    }

    /// Evaluates the farthest point on the shape along direction `dir` in local space.
    #[must_use]
    pub fn support(&self, dir: Vec3) -> Vec3 {
        match self {
            Self::Sphere { radius } => {
                let len = dir.length();
                if len > 1e-6 {
                    dir * (*radius / len)
                } else {
                    Vec3::new(0.0, *radius, 0.0)
                }
            }
            Self::Box { half_extents } => Vec3::new(
                if dir.x >= 0.0 {
                    half_extents.x
                } else {
                    -half_extents.x
                },
                if dir.y >= 0.0 {
                    half_extents.y
                } else {
                    -half_extents.y
                },
                if dir.z >= 0.0 {
                    half_extents.z
                } else {
                    -half_extents.z
                },
            ),
            Self::Capsule {
                radius,
                half_height,
            } => {
                let extreme_center = if dir.y >= 0.0 {
                    Vec3::new(0.0, *half_height, 0.0)
                } else {
                    Vec3::new(0.0, -*half_height, 0.0)
                };
                let len = dir.length();
                let sphere_offset = if len > 1e-6 {
                    dir * (*radius / len)
                } else {
                    Vec3::ZERO
                };
                extreme_center + sphere_offset
            }
            Self::ConvexHull(hull) => {
                let mut best_pt = Vec3::ZERO;
                let mut best_dot = f32::NEG_INFINITY;
                for &v in hull.points() {
                    let dot = v.dot(dir);
                    if dot > best_dot {
                        best_dot = dot;
                        best_pt = v;
                    }
                }
                best_pt
            }
            Self::TriangleMesh(mesh) => {
                let mut best_pt = Vec3::ZERO;
                let mut best_dot = f32::NEG_INFINITY;
                for &v in mesh.vertices() {
                    let dot = v.dot(dir);
                    if dot > best_dot {
                        best_dot = dot;
                        best_pt = v;
                    }
                }
                best_pt
            }
            Self::HeightField(hf) => {
                let (min, max) = hf.bounds();
                let points = [min, max];
                let mut best_pt = Vec3::ZERO;
                let mut best_dot = f32::NEG_INFINITY;
                for &v in &points {
                    let dot = v.dot(dir);
                    if dot > best_dot {
                        best_dot = dot;
                        best_pt = v;
                    }
                }
                best_pt
            }
            Self::Compound(comp) => {
                let (min, max) = comp.bounds();
                let points = [min, max];
                let mut best_pt = Vec3::ZERO;
                let mut best_dot = f32::NEG_INFINITY;
                for &v in &points {
                    let dot = v.dot(dir);
                    if dot > best_dot {
                        best_dot = dot;
                        best_pt = v;
                    }
                }
                best_pt
            }
            Self::Triangle { a, b, c } => {
                let mut best = *a;
                let mut best_dot = a.dot(dir);
                let d_b = b.dot(dir);
                if d_b > best_dot {
                    best = *b;
                    best_dot = d_b;
                }
                let d_c = c.dot(dir);
                if d_c > best_dot {
                    best = *c;
                }
                best
            }
        }
    }

    /// Casts a ray in local coordinates against this shape up to `max_distance`.
    ///
    /// Returns `(distance, normal)` if an intersection is found within `[0.0, max_distance]`.
    #[must_use]
    pub fn ray_cast(&self, ray: &Ray, max_distance: f32) -> Option<(f32, Vec3)> {
        match self {
            Self::Sphere { radius } => {
                let r = *radius;
                let s = ray.origin;
                let d = ray.direction;
                let b = s.dot(d);
                let c = s.dot(s) - r * r;

                if c < 0.0 {
                    // Ray origin is inside the sphere
                    return Some((0.0, -d));
                }

                if b > 0.0 {
                    // Ray points away from the sphere
                    return None;
                }

                let discr = b * b - c;
                if discr < 0.0 {
                    return None;
                }

                let t = -b - discr.sqrt();
                if t < 0.0 || t > max_distance {
                    return None;
                }

                let hit_point = ray.point_at(t);
                let normal = if r > 1e-6 { hit_point / r } else { -d };
                Some((t, normal))
            }
            Self::Box { half_extents } => {
                let mut lower = 0.0f32;
                let mut upper = max_distance;
                let mut hit_normal = -ray.direction;

                for i in 0..3 {
                    let d_i = ray.direction[i];
                    let o_i = ray.origin[i];
                    let h_i = half_extents[i];

                    let mut axis = Vec3::ZERO;
                    axis[i] = 1.0;

                    if d_i.abs() < 1e-8 {
                        if o_i < -h_i || o_i > h_i {
                            return None;
                        }
                    } else {
                        let mut t1 = (-h_i - o_i) / d_i;
                        let mut t2 = (h_i - o_i) / d_i;
                        let mut n1 = -axis;

                        if t1 > t2 {
                            std::mem::swap(&mut t1, &mut t2);
                            n1 = axis;
                        }

                        if t1 > lower {
                            lower = t1;
                            hit_normal = n1;
                        }
                        if t2 < upper {
                            upper = t2;
                        }

                        if upper < lower {
                            return None;
                        }
                    }
                }

                if lower < 0.0 {
                    Some((0.0, -ray.direction))
                } else if lower <= max_distance {
                    Some((lower, hit_normal))
                } else {
                    None
                }
            }
            Self::Capsule {
                radius,
                half_height,
            } => {
                let r = *radius;
                let hh = *half_height;

                let o = ray.origin;
                let d = ray.direction;

                let mut best_t = f32::INFINITY;
                let mut best_normal = Vec3::ZERO;

                // Check cylinder body
                let a = d.x * d.x + d.z * d.z;
                if a > 1e-8 {
                    let b = o.x * d.x + o.z * d.z;
                    let c = o.x * o.x + o.z * o.z - r * r;
                    let discr = b * b - a * c;
                    if discr >= 0.0 {
                        let sqrt_d = discr.sqrt();
                        let mut t = (-b - sqrt_d) / a;
                        if t < 0.0 {
                            t = (-b + sqrt_d) / a;
                        }
                        if (0.0..=max_distance).contains(&t) {
                            let y = o.y + t * d.y;
                            if (-hh..=hh).contains(&y) {
                                best_t = t;
                                best_normal =
                                    Vec3::new((o.x + t * d.x) / r, 0.0, (o.z + t * d.z) / r);
                            }
                        }
                    }
                }

                // Check top sphere cap at (0, hh, 0)
                let s_top = o - Vec3::new(0.0, hh, 0.0);
                let b_top = s_top.dot(d);
                let c_top = s_top.dot(s_top) - r * r;
                let discr_top = b_top * b_top - c_top;
                if discr_top >= 0.0 {
                    let t_top = -b_top - discr_top.sqrt();
                    if t_top >= 0.0 && t_top <= max_distance && t_top < best_t {
                        let pt = s_top + d * t_top;
                        if pt.y >= 0.0 {
                            best_t = t_top;
                            best_normal = pt.normalize();
                        }
                    }
                }

                // Check bottom sphere cap at (0, -hh, 0)
                let s_bot = o - Vec3::new(0.0, -hh, 0.0);
                let b_bot = s_bot.dot(d);
                let c_bot = s_bot.dot(s_bot) - r * r;
                let discr_bot = b_bot * b_bot - c_bot;
                if discr_bot >= 0.0 {
                    let t_bot = -b_bot - discr_bot.sqrt();
                    if t_bot >= 0.0 && t_bot <= max_distance && t_bot < best_t {
                        let pt = s_bot + d * t_bot;
                        if pt.y <= 0.0 {
                            best_t = t_bot;
                            best_normal = pt.normalize();
                        }
                    }
                }

                if best_t <= max_distance {
                    Some((best_t, best_normal))
                } else {
                    None
                }
            }
            // Box3D `b3IntersectRayTriangle`, from the front only.
            Self::Triangle { a, b, c } => {
                let translation = ray.direction * max_distance;
                let alpha =
                    crate::mesh::intersect_ray_triangle(ray.origin, translation, *a, *b, *c);
                (alpha < 1.0).then(|| {
                    let normal = crate::contact_solver::normalize((*b - *a).cross(*c - *a));
                    (alpha * max_distance, normal)
                })
            }
            // b3RayCastMesh.
            Self::TriangleMesh(mesh) => mesh
                .ray_cast(ray.origin, ray.direction * max_distance, 1.0)
                .map(|hit| (hit.fraction * max_distance, hit.normal)),
            // b3RayCastHeightField.
            Self::HeightField(hf) => hf
                .ray_cast(ray.origin, ray.direction * max_distance, 1.0)
                .map(|hit| (hit.fraction * max_distance, hit.normal)),
            // b3RayCastCompound.
            Self::Compound(comp) => comp
                .ray_cast(ray.origin, ray.direction * max_distance, 1.0)
                .map(|hit| (hit.fraction * max_distance, hit.normal)),
            // b3RayCastHull: a ray starting inside hits at once, without a normal.
            Self::ConvexHull(hull) => hull
                .ray_cast(ray.origin, ray.direction * max_distance, 1.0)
                .map(|(fraction, normal)| (fraction * max_distance, normal)),
        }
    }

    /// Casts a ray in local coordinates against this shape up to `max_distance`, returning
    /// `(distance, normal, triangle_or_child_index)`.
    #[must_use]
    pub fn ray_cast_detailed(&self, ray: &Ray, max_distance: f32) -> Option<(f32, Vec3, usize)> {
        match self {
            Self::TriangleMesh(mesh) => mesh
                .ray_cast(ray.origin, ray.direction * max_distance, 1.0)
                .map(|hit| (hit.fraction * max_distance, hit.normal, hit.triangle)),
            Self::HeightField(hf) => hf
                .ray_cast(ray.origin, ray.direction * max_distance, 1.0)
                .map(|hit| (hit.fraction * max_distance, hit.normal, hit.triangle)),
            Self::Compound(comp) => comp
                .ray_cast(ray.origin, ray.direction * max_distance, 1.0)
                .map(|hit| (hit.fraction * max_distance, hit.normal, hit.child_index)),
            _ => self.ray_cast(ray, max_distance).map(|(d, n)| (d, n, 0)),
        }
    }
}

/// Box3D `b3AABB_Transform`: the world box around local bounds `(lower, upper)` at a pose.
fn transform_bounds((lower, upper): (Vec3, Vec3), position: Vec3, rotation: Quat) -> Aabb {
    let (center, extents) = ((lower + upper) * 0.5, (upper - lower) * 0.5);
    let r = Mat3::from_quat(rotation);
    let world_extents =
        (r.x_axis * extents.x).abs() + (r.y_axis * extents.y).abs() + (r.z_axis * extents.z).abs();
    let world_center = position + rotation * center;
    Aabb {
        min: world_center - world_extents,
        max: world_center + world_extents,
    }
}
