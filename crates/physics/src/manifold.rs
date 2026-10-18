//! Box3D's convex manifold builders (`convex_manifold.c`, `manifold.c`).
//!
//! Each builder collides shape A with shape B given in A's frame and returns a
//! [`LocalManifold`] in A's frame: a normal from A to B and up to four points midway between
//! the surfaces, with their separations (negative for overlap) and the feature pair that made
//! them. Points are kept up to [`SPECULATIVE_DISTANCE`] apart, except for spheres against
//! spheres or capsules, which Box3D collides only when they touch. Hulls (boxes) find the axis
//! of least penetration with Box3D's separating axis test, cached in a [`SatCache`] between
//! steps, clip the incident face against the reference face and reduce the points to four;
//! capsules clip their segment against a face or a parallel capsule for two points.

use crate::contact_solver::{normalize, LINEAR_SLOP, SPECULATIVE_DISTANCE};
use crate::distance::{
    closest_point_on_segment, line_distance, segment_distance, shape_distance, ShapeProxy,
    SimplexCache,
};
use crate::hull::{HullView, Plane};
use crate::math::{lerp, Transform};
use korppi_math::{Mat3, Vec3};
use serde::{Deserialize, Serialize};

/// The most points a manifold keeps (`B3_MAX_MANIFOLD_POINTS`).
pub(crate) const MAX_POINTS: usize = 4;

/// The most points a clipped polygon holds (`B3_MAX_CLIP_POINTS`).
pub(crate) const MAX_CLIP_POINTS: usize = 64;

/// Box3D's relative tolerance for parallel edges (`B3_PARALLEL_EDGE_TOL`).
const PARALLEL_EDGE_TOL: f32 = 0.005;

/// The shortest capsule segment (`B3_MIN_CAPSULE_LENGTH`).
const MIN_CAPSULE_LENGTH: f32 = LINEAR_SLOP;

/// Box3D's threshold for vectors too short to use, `1000 * FLT_MIN`.
const MIN_LENGTH_SQUARED: f32 = 1000.0 * f32::MIN_POSITIVE;

pub(crate) const SHAPE_A: u8 = 0;
pub(crate) const SHAPE_B: u8 = 1;

/// The two edges whose crossing made a contact point (Box3D `b3FeaturePair`), for matching
/// points between steps.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FeaturePair {
    owner1: u8,
    index1: u8,
    owner2: u8,
    index2: u8,
}

impl FeaturePair {
    /// The pair of a lone point (Box3D `b3FeaturePair_single`).
    pub const SINGLE: Self = Self {
        owner1: 0,
        index1: 0,
        owner2: 0,
        index2: 0,
    };

    pub(crate) fn new(owner1: u8, index1: usize, owner2: u8, index2: usize) -> Self {
        Self {
            owner1,
            index1: index1 as u8,
            owner2,
            index2: index2 as u8,
        }
    }

    /// Box3D `b3FlipPair`: the same pair seen from the other shape's reference face.
    pub(crate) fn flip(self) -> Self {
        Self {
            owner1: 1 - self.owner2,
            index1: self.index2,
            owner2: 1 - self.owner1,
            index2: self.index1,
        }
    }

    /// Box3D `b3MakeFeatureId`.
    pub fn id(self) -> u32 {
        (u32::from(self.owner1) << 24)
            | (u32::from(self.index1) << 16)
            | (u32::from(self.owner2) << 8)
            | u32::from(self.index2)
    }
}

/// A manifold point in shape A's frame (Box3D `b3LocalManifoldPoint`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct LocalPoint {
    pub point: Vec3,
    pub separation: f32,
    pub pair: FeaturePair,
}

/// A manifold in shape A's frame (Box3D `b3LocalManifold`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct LocalManifold {
    pub normal: Vec3,
    pub points: [LocalPoint; MAX_POINTS],
    pub count: usize,
}

impl LocalManifold {
    pub fn points(&self) -> &[LocalPoint] {
        &self.points[..self.count]
    }

    fn single(normal: Vec3, point: Vec3, separation: f32, pair: FeaturePair) -> Self {
        let mut manifold = Self {
            normal,
            count: 1,
            ..Self::default()
        };
        manifold.points[0] = LocalPoint {
            point,
            separation,
            pair,
        };
        manifold
    }

    fn pair_of(normal: Vec3, first: LocalPoint, second: LocalPoint) -> Self {
        let mut manifold = Self {
            normal,
            count: 2,
            ..Self::default()
        };
        manifold.points[0] = first;
        manifold.points[1] = second;
        manifold
    }
}

/// The separating axis test's memory between steps (Box3D `b3SATCache`): the axis that
/// separated or touched last, and its separation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SatCache {
    /// The separation when the cache was written; negative for overlap.
    pub separation: f32,
    /// The kind of axis: none, a face of A, a face of B or an edge pair.
    pub kind: u8,
    /// The feature on shape A.
    pub index_a: u8,
    /// The feature on shape B.
    pub index_b: u8,
    /// Whether the last query reused the cache.
    pub hit: u8,
}

pub(crate) const AXIS_INVALID: u8 = 0;
pub(crate) const AXIS_FACE_A: u8 = 1;
pub(crate) const AXIS_FACE_B: u8 = 2;
pub(crate) const AXIS_EDGE_PAIR: u8 = 3;
/// A triangle seen from behind, with the hull centre's offset as the separation.
pub(crate) const AXIS_BACKSIDE: u8 = 4;

/// A sphere in its shape's frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

/// A capsule in its shape's frame (Box3D `b3Capsule`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Capsule {
    pub center1: Vec3,
    pub center2: Vec3,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ClipVertex {
    pub position: Vec3,
    pub separation: f32,
    pub pair: FeaturePair,
}

#[derive(Debug, Clone, Copy)]
struct SeparatingAxis {
    normal: Vec3,
    separation: f32,
    index_a: i32,
    index_b: i32,
}

impl SeparatingAxis {
    const NONE: Self = Self {
        normal: Vec3::ZERO,
        separation: f32::NEG_INFINITY,
        index_a: -1,
        index_b: -1,
    };
}

#[derive(Debug, Clone, Copy)]
struct AxisQuery {
    face_a: SeparatingAxis,
    face_b: SeparatingAxis,
    edge: SeparatingAxis,
    separated: u8,
}

/// Box3D `b3GetLengthAndNormalize`.
fn length_and_normalize(v: Vec3) -> (f32, Vec3) {
    let length_squared = v.x * v.x + v.y * v.y + v.z * v.z;
    if length_squared > MIN_LENGTH_SQUARED {
        let length = length_squared.sqrt();
        let s = 1.0 / length;
        (length, Vec3::new(s * v.x, s * v.y, s * v.z))
    } else {
        (0.0, Vec3::ZERO)
    }
}

/// Box3D `b3ArbitraryPerp`: a unit vector perpendicular to the unit vector `v`.
fn arbitrary_perp(v: Vec3) -> Vec3 {
    let p = if v.x < -0.5 || 0.5 < v.x {
        let (a, b) = (0.67, -0.42);
        Vec3::new(a * v.y + b * v.z, -a * v.x, -b * v.x)
    } else if v.y < -0.5 || 0.5 < v.y {
        let (a, c) = (0.67, -0.42);
        Vec3::new(a * v.y, -a * v.x + c * v.z, -c * v.y)
    } else {
        let (a, b) = (0.67, -0.42);
        Vec3::new(a * v.z, b * v.z, -a * v.x - b * v.y)
    };
    normalize(p)
}

// ---------------------------------------------------------------- spheres and capsules

/// Box3D `b3CollideSpheres`: touching spheres only, no speculative margin.
pub(crate) fn collide_spheres(a: &Sphere, b: &Sphere, xf: &Transform) -> LocalManifold {
    let center1 = a.center;
    let center2 = xf.apply(b.center);
    let total_radius = a.radius + b.radius;
    let offset = center2 - center1;
    let distance_squared = offset.length_squared();
    if distance_squared > total_radius * total_radius {
        return LocalManifold::default();
    }
    let distance = distance_squared.sqrt();
    let normal = if distance * distance > MIN_LENGTH_SQUARED {
        offset * (1.0 / distance)
    } else {
        Vec3::Y
    };
    // Midway between the surfaces.
    let point = ((center1 + normal * a.radius) + center2 - normal * b.radius) * 0.5;
    LocalManifold::single(normal, point, distance - total_radius, FeaturePair::SINGLE)
}

/// Box3D `b3CollideCapsuleAndSphere`: touching shapes only, no speculative margin.
pub(crate) fn collide_capsule_and_sphere(a: &Capsule, b: &Sphere, xf: &Transform) -> LocalManifold {
    let center = xf.apply(b.center);
    let total_radius = b.radius + a.radius;
    let closest = closest_point_on_segment(a.center1, a.center2, center);
    let offset = center - closest;
    let distance_squared = offset.length_squared();
    if distance_squared > total_radius * total_radius {
        return LocalManifold::default();
    }
    let distance = distance_squared.sqrt();
    let normal = if distance * distance > MIN_LENGTH_SQUARED {
        offset * (1.0 / distance)
    } else {
        Vec3::Y
    };
    let point = ((center - normal * b.radius) + closest + normal * a.radius) * 0.5;
    LocalManifold::single(normal, point, distance - total_radius, FeaturePair::SINGLE)
}

/// Box3D `b3ClipSegment`: keeps the part of a segment behind `plane`.
fn clip_segment(segment: &mut [ClipVertex; 2], plane: &Plane) -> usize {
    let (vertex1, vertex2) = (segment[0], segment[1]);
    let distance1 = plane.separation(vertex1.position);
    let distance2 = plane.separation(vertex2.position);
    let mut count = 0;
    if distance1 <= 0.0 {
        segment[count] = vertex1;
        count += 1;
    }
    if distance2 <= 0.0 {
        segment[count] = vertex2;
        count += 1;
    }
    if distance1 * distance2 < 0.0 {
        let t = distance1 / (distance1 - distance2);
        segment[count].position = vertex1.position * (1.0 - t) + vertex2.position * t;
        segment[count].pair = if distance1 > 0.0 {
            vertex1.pair
        } else {
            vertex2.pair
        };
        count += 1;
    }
    count
}

/// Box3D `b3CollideCapsules`: the closest points of the segments, or two points clipped
/// along nearly parallel capsules that touch at both.
pub(crate) fn collide_capsules(a: &Capsule, b: &Capsule, xf: &Transform) -> LocalManifold {
    let (a1, a2) = (a.center1, a.center2);
    let b1 = xf.apply(b.center1);
    let b2 = xf.apply(b.center2);
    let radius = a.radius + b.radius;
    let max_distance = radius + SPECULATIVE_DISTANCE;

    let result = segment_distance(a1, a2, b1, b2);
    let offset = result.point2 - result.point1;
    let distance_squared = offset.length_squared();
    let min_distance = 0.01 * LINEAR_SLOP;
    if distance_squared > max_distance * max_distance
        || distance_squared < min_distance * min_distance
    {
        return LocalManifold::default();
    }

    let (length_a, edge_a) = length_and_normalize(a2 - a1);
    if length_a < MIN_CAPSULE_LENGTH {
        return LocalManifold::default();
    }
    let (length_b, edge_b) = length_and_normalize(b2 - b1);
    if length_b < MIN_CAPSULE_LENGTH {
        return LocalManifold::default();
    }

    // Nearly parallel: clip segment B against the side planes of segment A.
    let alpha_tolerance = 0.05;
    if edge_a.cross(edge_b).length_squared() < alpha_tolerance * alpha_tolerance {
        let planes = [
            Plane {
                normal: -edge_a,
                offset: -edge_a.dot(a.center1),
            },
            Plane {
                normal: edge_a,
                offset: edge_a.dot(a.center2),
            },
        ];
        let mut vertices = [
            ClipVertex {
                position: b1,
                separation: 0.0,
                pair: FeaturePair::new(SHAPE_A, 0, SHAPE_A, 0),
            },
            ClipVertex {
                position: b2,
                separation: 0.0,
                pair: FeaturePair::new(SHAPE_A, 1, SHAPE_A, 1),
            },
        ];
        let mut count = clip_segment(&mut vertices, &planes[0]);
        if count == 2 {
            count = clip_segment(&mut vertices, &planes[1]);
        }
        if count == 2 {
            let closest1 = closest_point_on_segment(a1, a2, vertices[0].position);
            let closest2 = closest_point_on_segment(a1, a2, vertices[1].position);
            let distance1 = (vertices[0].position - closest1).length();
            let distance2 = (vertices[1].position - closest2).length();
            if distance1 <= radius && distance2 <= radius {
                if distance1 < min_distance || distance2 < min_distance {
                    return LocalManifold::default();
                }
                let normal1 = (vertices[0].position - closest1) * (1.0 / distance1);
                let normal2 = (vertices[1].position - closest2) * (1.0 / distance2);
                let normal = normalize(normal1 + normal2);
                let point1 = ((vertices[0].position + normal1 * a.radius) + closest1
                    - normal * b.radius)
                    * 0.5;
                let point2 = ((vertices[1].position + normal2 * a.radius) + closest2
                    - normal * b.radius)
                    * 0.5;
                return LocalManifold::pair_of(
                    normal,
                    LocalPoint {
                        point: point1,
                        separation: distance1 - radius,
                        pair: vertices[0].pair,
                    },
                    LocalPoint {
                        point: point2,
                        separation: distance2 - radius,
                        pair: vertices[1].pair,
                    },
                );
            }
        }
    }

    let (distance, normal) = length_and_normalize(offset);
    let point = ((result.point1 + normal * a.radius) + result.point2 - normal * b.radius) * 0.5;
    LocalManifold::single(normal, point, distance - radius, FeaturePair::SINGLE)
}

// ---------------------------------------------------------------- hulls and round shapes

/// Box3D `b3CollideHullAndSphere`: GJK for the closest points, the face of least penetration
/// when the centre is inside.
pub(crate) fn collide_hull_and_sphere(
    hull: &HullView<'_>,
    sphere: &Sphere,
    xf: &Transform,
    cache: &mut SimplexCache,
) -> LocalManifold {
    let center = xf.apply(sphere.center);
    let centers = [center];
    let output = shape_distance(
        &ShapeProxy {
            points: hull.points,
            radius: 0.0,
        },
        &ShapeProxy {
            points: &centers,
            radius: 0.0,
        },
        &Transform::IDENTITY,
        false,
        cache,
    );
    let radius = sphere.radius;
    if output.distance > radius + SPECULATIVE_DISTANCE {
        *cache = SimplexCache::default();
        return LocalManifold::default();
    }

    let (normal, separation) = if output.distance > 100.0 * f32::EPSILON {
        // Shallow penetration.
        (
            normalize(output.point_b - output.point_a),
            output.distance - radius,
        )
    } else {
        // Deep penetration: the face the centre is least deep behind.
        let mut best = 0;
        let mut best_distance = -f32::MAX;
        for (index, plane) in hull.planes.iter().enumerate() {
            let distance = plane.separation(center);
            if distance > best_distance {
                best = index;
                best_distance = distance;
            }
        }
        (hull.planes[best].normal, best_distance - radius)
    };
    // The centre projected onto the hull, and the deepest point of the sphere.
    let c_a = center + normal * (0.0 - (center - output.point_a).dot(normal));
    let c_b = center - normal * radius;
    LocalManifold::single(normal, lerp(c_a, c_b, 0.5), separation, FeaturePair::SINGLE)
}

/// Box3D `b3ClipSegmentToHullFace`: clips a segment against the side planes of a face.
fn clip_segment_to_hull_face(
    segment: &mut [ClipVertex; 2],
    hull: &HullView<'_>,
    face: usize,
) -> usize {
    let ref_plane = hull.planes[face];
    let first = hull.face_edges[face] as usize;
    let mut edge_index = first;
    loop {
        let edge = hull.edges[edge_index];
        let next_index = edge.next as usize;
        let next = hull.edges[next_index];
        let vertex1 = hull.points[edge.origin as usize];
        let vertex2 = hull.points[next.origin as usize];
        let tangent = normalize(vertex2 - vertex1);
        let binormal = tangent.cross(ref_plane.normal);
        if clip_segment(segment, &Plane::from_normal_and_point(binormal, vertex1)) < 2 {
            return 0;
        }
        edge_index = next_index;
        if edge_index == first {
            return 2;
        }
    }
}

/// Box3D `b3QueryFaceDirectionHullAndCapsule`.
fn query_face_hull_and_capsule(
    hull: &HullView<'_>,
    capsule: &Capsule,
    xf: &Transform,
) -> SeparatingAxis {
    let points = [xf.apply(capsule.center1), xf.apply(capsule.center2)];
    let mut best = SeparatingAxis {
        separation: -f32::MAX,
        ..SeparatingAxis::NONE
    };
    for (face, plane) in hull.planes.iter().enumerate() {
        let vertex = crate::distance::support_index(&points, -plane.normal);
        let separation = plane.separation(points[vertex]);
        if separation > best.separation {
            best = SeparatingAxis {
                normal: plane.normal,
                separation,
                index_a: face as i32,
                index_b: vertex as i32,
            };
        }
    }
    best
}

/// Box3D `b3QueryEdgeDirectionHullAndCapsule`.
fn query_edges_hull_and_capsule(
    hull: &HullView<'_>,
    capsule: &Capsule,
    xf: &Transform,
) -> SeparatingAxis {
    let p_a = xf.apply(capsule.center1);
    let q_a = xf.apply(capsule.center2);
    let e_a = q_a - p_a;
    let mut best = SeparatingAxis {
        separation: -f32::MAX,
        ..SeparatingAxis::NONE
    };
    let tolerance_squared = PARALLEL_EDGE_TOL * PARALLEL_EDGE_TOL;
    for index in (0..hull.edges.len()).step_by(2) {
        let edge = hull.edges[index];
        let twin = hull.edges[index + 1];
        let q_b = hull.points[twin.origin as usize];
        let u_b = hull.planes[edge.face as usize].normal;
        let v_b = hull.planes[twin.face as usize].normal;
        // A capsule's edge is a great circle on the Gauss map through the origin, so the arc
        // test is a plane test.
        let cba = u_b.dot(e_a);
        let dba = v_b.dot(e_a);
        if cba * dba < 0.0 {
            if (cba * cba).max(dba * dba) < tolerance_squared * e_a.length_squared() {
                continue;
            }
            let t = cba / (cba - dba);
            let axis = normalize(lerp(u_b, v_b, t));
            let separation = axis.dot(q_a - q_b);
            if separation > best.separation {
                best = SeparatingAxis {
                    normal: axis,
                    separation,
                    index_a: 0,
                    index_b: index as i32,
                };
            }
        }
    }
    best
}

/// The two points of a capsule segment `segment` clipped to reference face `plane`, half-way
/// between the capsule's surface and the face.
fn capsule_face_points(segment: &[ClipVertex; 2], plane: &Plane, radius: f32) -> LocalManifold {
    let distance1 = plane.separation(segment[0].position);
    let distance2 = plane.separation(segment[1].position);
    let normal = plane.normal;
    LocalManifold::pair_of(
        normal,
        LocalPoint {
            point: segment[0].position - normal * (0.5 * (distance1 + radius)),
            separation: distance1 - radius,
            pair: segment[0].pair,
        },
        LocalPoint {
            point: segment[1].position - normal * (0.5 * (distance2 + radius)),
            separation: distance2 - radius,
            pair: segment[1].pair,
        },
    )
}

fn capsule_segment(capsule: &Capsule, xf: &Transform) -> [ClipVertex; 2] {
    [
        ClipVertex {
            position: xf.apply(capsule.center1),
            separation: 0.0,
            pair: FeaturePair::new(SHAPE_A, 0, SHAPE_A, 0),
        },
        ClipVertex {
            position: xf.apply(capsule.center2),
            separation: 0.0,
            pair: FeaturePair::new(SHAPE_A, 1, SHAPE_A, 1),
        },
    ]
}

/// Box3D `b3BuildHullFaceAndCapsuleContact`.
fn hull_face_and_capsule_contact(
    hull: &HullView<'_>,
    capsule: &Capsule,
    xf: &Transform,
    query: &SeparatingAxis,
) -> LocalManifold {
    let face = query.index_a as usize;
    let ref_plane = hull.planes[face];
    let mut segment = capsule_segment(capsule, xf);
    if clip_segment_to_hull_face(&mut segment, hull, face) < 2 {
        return LocalManifold::default();
    }
    let distance1 = ref_plane.separation(segment[0].position);
    let distance2 = ref_plane.separation(segment[1].position);
    if distance1 <= SPECULATIVE_DISTANCE || distance2 <= SPECULATIVE_DISTANCE {
        capsule_face_points(&segment, &ref_plane, capsule.radius)
    } else {
        LocalManifold::default()
    }
}

/// Box3D `b3BuildHullAndCapsuleEdgeContact`.
fn hull_and_capsule_edge_contact(
    hull: &HullView<'_>,
    capsule: &Capsule,
    xf: &Transform,
    query: &SeparatingAxis,
) -> LocalManifold {
    let p_c = xf.apply(capsule.center1);
    let q_c = xf.apply(capsule.center2);
    let edge = hull.edges[query.index_b as usize];
    let twin = hull.edges[edge.twin as usize];
    let p_h = hull.points[edge.origin as usize];
    let q_h = hull.points[twin.origin as usize];
    let normal = query.normal;
    let result = line_distance(p_h, q_h - p_h, p_c, q_c - p_c);
    if !result.is_within_segments() {
        return LocalManifold::default();
    }
    let point = ((result.point1 - normal * capsule.radius) + result.point2) * 0.5;
    let separation = normal.dot(result.point2 - result.point1);
    LocalManifold::single(
        normal,
        point,
        separation - capsule.radius,
        FeaturePair::new(
            SHAPE_A,
            query.index_a as usize,
            SHAPE_B,
            query.index_b as usize,
        ),
    )
}

/// Box3D `b3CollideHullAndCapsule`: two points when the capsule lies along a face, else the
/// closest points; separating axes when the segment is inside.
pub(crate) fn collide_hull_and_capsule(
    hull: &HullView<'_>,
    capsule: &Capsule,
    xf: &Transform,
    cache: &mut SimplexCache,
) -> LocalManifold {
    let centers = [capsule.center1, capsule.center2];
    let output = shape_distance(
        &ShapeProxy {
            points: hull.points,
            radius: 0.0,
        },
        &ShapeProxy {
            points: &centers,
            radius: 0.0,
        },
        xf,
        false,
        cache,
    );
    let radius = capsule.radius;
    if output.distance > radius + SPECULATIVE_DISTANCE {
        *cache = SimplexCache::default();
        return LocalManifold::default();
    }

    if output.distance > 100.0 * f32::EPSILON {
        // Shallow penetration.
        let delta = output.normal;
        let face = hull.support_face(delta);
        let ref_plane = hull.planes[face];
        // Two points when the closest points line up with the face normal.
        if ref_plane.normal.dot(delta).abs() > 0.998 {
            let mut segment = capsule_segment(capsule, xf);
            if clip_segment_to_hull_face(&mut segment, hull, face) == 2 {
                let distance1 = ref_plane.separation(segment[0].position);
                let distance2 = ref_plane.separation(segment[1].position);
                let limit = radius + SPECULATIVE_DISTANCE;
                if distance1 <= limit || distance2 <= limit {
                    return capsule_face_points(&segment, &ref_plane, radius);
                }
            }
        }
        let point = ((output.point_a - delta * radius) + output.point_b) * 0.5;
        return LocalManifold::single(delta, point, output.distance - radius, FeaturePair::SINGLE);
    }

    // Deep penetration.
    let face_query = query_face_hull_and_capsule(hull, capsule, xf);
    if face_query.separation > radius {
        return LocalManifold::default();
    }
    let edge_query = query_edges_hull_and_capsule(hull, capsule, xf);
    if edge_query.separation > radius {
        return LocalManifold::default();
    }

    let mut manifold = hull_face_and_capsule_contact(hull, capsule, xf, &face_query);
    let face_separation = if manifold.count == 2 {
        manifold.points[0]
            .separation
            .min(manifold.points[1].separation)
    } else {
        face_query.separation - radius
    };
    if edge_query.index_a < 0 {
        return manifold;
    }
    // Edge contact if the face contact failed or the edge is clearly better.
    let edge_separation = edge_query.separation - radius;
    if manifold.count == 0 || edge_separation > face_separation + LINEAR_SLOP {
        manifold = hull_and_capsule_edge_contact(hull, capsule, xf, &edge_query);
    }
    manifold
}

// ---------------------------------------------------------------- hull against hull

/// Box3D's four-point `b3ReduceManifoldPoints`: the deepest extreme point, the farthest from
/// it, the largest triangle and the point adding the most area. `manifold.normal` must be set.
fn reduce_points(manifold: &mut LocalManifold, points: &mut [LocalPoint]) {
    let mut count = points.len();
    if count <= MAX_POINTS {
        manifold.points[..count].copy_from_slice(points);
        manifold.count = count;
        return;
    }

    let normal = manifold.normal;
    let tolerance_squared = SPECULATIVE_DISTANCE * SPECULATIVE_DISTANCE;
    // A pecking order that keeps the choice stable across steps.
    let bias = 0.95;

    // 1: the extreme touching point.
    let search_direction = arbitrary_perp(normal);
    let mut best_index = None;
    let mut best_score = -f32::MAX;
    for (index, p) in points[..count].iter().enumerate() {
        if p.separation > SPECULATIVE_DISTANCE {
            continue;
        }
        let score = -p.separation + search_direction.dot(p.point);
        if bias * score > best_score {
            best_index = Some(index);
            best_score = score;
        }
    }
    let Some(best) = best_index else {
        manifold.count = 0;
        return;
    };
    manifold.points[0] = points[best];
    manifold.count = 1;
    points[best] = points[count - 1];
    count -= 1;
    let a = manifold.points[0].point;

    // 2: the farthest point in the plane.
    let mut best_score = 0.0;
    let mut best_index = None;
    for (index, p) in points[..count].iter().enumerate() {
        let d = p.point - a;
        let v = d - normal * d.dot(normal);
        let depth = (-p.separation).max(0.0);
        let score = v.length_squared() + 4.0 * depth * depth;
        if bias * score > best_score {
            best_score = score;
            best_index = Some(index);
        }
    }
    if best_score < tolerance_squared {
        return;
    }
    let Some(best) = best_index else {
        return;
    };
    manifold.points[1] = points[best];
    manifold.count = 2;
    points[best] = points[count - 1];
    count -= 1;
    let b = manifold.points[1].point;

    // 3: the largest triangle.
    let mut best_score = tolerance_squared;
    let mut best_index = None;
    let mut best_signed_area = 0.0;
    let ba = b - a;
    for (index, p) in points[..count].iter().enumerate() {
        let signed_area = normal.dot(ba.cross(p.point - a));
        let score = signed_area.abs();
        if bias * score >= best_score {
            best_score = score;
            best_index = Some(index);
            best_signed_area = signed_area;
        }
    }
    let Some(best) = best_index else {
        return;
    };
    manifold.points[2] = points[best];
    manifold.count = 3;
    points[best] = points[count - 1];
    count -= 1;
    let c = manifold.points[2].point;

    // 4: the point adding the most area outside the triangle.
    let mut best_score = tolerance_squared;
    let mut best_index = None;
    let sign = if best_signed_area < 0.0 { -1.0 } else { 1.0 };
    for (index, p) in points[..count].iter().enumerate() {
        let u1 = sign * normal.dot((p.point - a).cross(ba));
        let u2 = sign * normal.dot((p.point - b).cross(c - b));
        let u3 = sign * normal.dot((p.point - c).cross(a - c));
        let score = u1.max(u2.max(u3));
        if bias * score > best_score {
            best_score = score;
            best_index = Some(index);
        }
    }
    if let Some(best) = best_index {
        manifold.points[3] = points[best];
        manifold.count = 4;
    }
}

/// Box3D `b3FindIncidentFace`: of the edges at `vertex`, the one most perpendicular to
/// `ref_normal`, and of its two faces the more anti-parallel one.
pub(crate) fn find_incident_face(hull: &HullView<'_>, ref_normal: Vec3, vertex: usize) -> usize {
    let first = hull.vertex_edges[vertex] as usize;
    let mut edge_index = first;
    let origin = hull.points[hull.edges[first].origin as usize];
    let mut min_edge = first;
    let mut min_projection = f32::MAX;
    loop {
        let edge = hull.edges[edge_index];
        let twin = hull.edges[edge.twin as usize];
        let axis = normalize(hull.points[twin.origin as usize] - origin);
        let projection = axis.dot(ref_normal).abs();
        if projection < min_projection {
            min_edge = edge_index;
            min_projection = projection;
        }
        edge_index = twin.next as usize;
        if edge_index == first {
            break;
        }
    }
    let edge = hull.edges[min_edge];
    let face1 = edge.face as usize;
    let face2 = hull.edges[edge.twin as usize].face as usize;
    if hull.planes[face1].normal.dot(ref_normal) < hull.planes[face2].normal.dot(ref_normal) {
        face1
    } else {
        face2
    }
}

/// Box3D `b3ClipPolygon`: Sutherland–Hodgman clipping of a polygon against `clip_plane`,
/// labelling new points with `edge` and measuring them from `ref_plane`.
pub(crate) fn clip_polygon(
    out: &mut [ClipVertex; MAX_CLIP_POINTS],
    polygon: &[ClipVertex],
    clip_plane: &Plane,
    edge: usize,
    ref_plane: &Plane,
) -> usize {
    let mut count = 0;
    let mut push = |out: &mut [ClipVertex; MAX_CLIP_POINTS], vertex: ClipVertex| {
        if count < MAX_CLIP_POINTS {
            out[count] = vertex;
            count += 1;
        }
    };
    let mut vertex1 = polygon[polygon.len() - 1];
    let mut distance1 = clip_plane.separation(vertex1.position);
    for &vertex2 in polygon {
        let distance2 = clip_plane.separation(vertex2.position);
        if distance1 <= 0.0 && distance2 <= 0.0 {
            push(out, vertex2);
        } else if distance1 <= 0.0 && distance2 > 0.0 {
            let fraction = distance1 / (distance1 - distance2);
            let position = vertex1.position + (vertex2.position - vertex1.position) * fraction;
            let mut pair = vertex2.pair;
            pair.owner2 = SHAPE_A;
            pair.index2 = edge as u8;
            push(
                out,
                ClipVertex {
                    position,
                    separation: ref_plane.separation(position),
                    pair,
                },
            );
        } else if distance2 <= 0.0 && distance1 > 0.0 {
            let fraction = distance1 / (distance1 - distance2);
            let position = vertex1.position + (vertex2.position - vertex1.position) * fraction;
            let mut pair = vertex1.pair;
            pair.owner1 = SHAPE_A;
            pair.index1 = edge as u8;
            push(
                out,
                ClipVertex {
                    position,
                    separation: ref_plane.separation(position),
                    pair,
                },
            );
            push(out, vertex2);
        }
        vertex1 = vertex2;
        distance1 = distance2;
    }
    count
}

/// Box3D `b3BuildPolygon`: face `face` of `hull` in the other frame through `xf`.
fn build_polygon(
    out: &mut [ClipVertex; MAX_CLIP_POINTS],
    xf: &Transform,
    hull: &HullView<'_>,
    face: usize,
    ref_plane: &Plane,
) -> usize {
    let matrix = Mat3::from_quat(xf.q);
    let first = hull.face_edges[face] as usize;
    let mut edge_index = first;
    let mut count = 0;
    loop {
        let next_index = hull.edges[edge_index].next as usize;
        let next = hull.edges[next_index];
        let position = matrix * hull.points[next.origin as usize] + xf.p;
        out[count] = ClipVertex {
            position,
            separation: ref_plane.separation(position),
            pair: FeaturePair::new(SHAPE_B, edge_index, SHAPE_B, next_index),
        };
        count += 1;
        edge_index = next_index;
        if edge_index == first || count == MAX_CLIP_POINTS {
            return count;
        }
    }
}

/// Box3D `b3BuildFaceAContact`: the incident face of B clipped to reference face
/// `query.index_a` of A.
fn build_face_a_contact(
    manifold: &mut LocalManifold,
    hull_a: &HullView<'_>,
    hull_b: &HullView<'_>,
    xf: &Transform,
    query: &SeparatingAxis,
    cache: &mut SatCache,
) -> bool {
    let ref_face = query.index_a as usize;
    let ref_plane = hull_a.planes[ref_face];
    let ref_normal_in_b = crate::math::inv_rotate(xf.q, ref_plane.normal);
    let inc_face = find_incident_face(hull_b, ref_normal_in_b, query.index_b as usize);

    let mut buffer1 = [ClipVertex::default(); MAX_CLIP_POINTS];
    let mut buffer2 = [ClipVertex::default(); MAX_CLIP_POINTS];
    let mut count = build_polygon(&mut buffer1, xf, hull_b, inc_face, &ref_plane);
    let (mut input, mut output) = (&mut buffer1, &mut buffer2);

    let first = hull_a.face_edges[ref_face] as usize;
    let mut edge_index = first;
    loop {
        let edge = hull_a.edges[edge_index];
        let next_index = edge.next as usize;
        let next = hull_a.edges[next_index];
        let vertex1 = hull_a.points[edge.origin as usize];
        let vertex2 = hull_a.points[next.origin as usize];
        let tangent = normalize(vertex2 - vertex1);
        let binormal = tangent.cross(ref_plane.normal);
        let clip_plane = Plane::from_normal_and_point(binormal, vertex1);
        count = clip_polygon(output, &input[..count], &clip_plane, edge_index, &ref_plane);
        std::mem::swap(&mut input, &mut output);
        if count < 3 {
            *cache = SatCache::default();
            return false;
        }
        edge_index = next_index;
        if edge_index == first {
            break;
        }
    }

    let mut points = [LocalPoint::default(); MAX_CLIP_POINTS];
    let mut min_separation = f32::MAX;
    manifold.normal = ref_plane.normal;
    for (point, clip) in points.iter_mut().zip(&input[..count]) {
        // Half-way, so the points stay put when the reference face swaps from A to B.
        *point = LocalPoint {
            point: clip.position - ref_plane.normal * (0.5 * clip.separation),
            separation: clip.separation,
            pair: clip.pair,
        };
        min_separation = min_separation.min(clip.separation);
    }
    if min_separation >= SPECULATIVE_DISTANCE {
        *cache = SatCache::default();
        return false;
    }
    reduce_points(manifold, &mut points[..count]);
    *cache = SatCache {
        separation: min_separation,
        kind: AXIS_FACE_A,
        index_a: query.index_a as u8,
        index_b: query.index_b as u8,
        hit: cache.hit,
    };
    true
}

/// Box3D `b3BuildFaceBContact`: reference face `query.index_b` of B, results back in A's
/// frame.
fn build_face_b_contact(
    manifold: &mut LocalManifold,
    hull_a: &HullView<'_>,
    hull_b: &HullView<'_>,
    xf: &Transform,
    query: &SeparatingAxis,
    cache: &mut SatCache,
) -> bool {
    let flipped = SeparatingAxis {
        normal: -query.normal,
        separation: query.separation,
        index_a: query.index_b,
        index_b: query.index_a,
    };
    if !build_face_a_contact(manifold, hull_b, hull_a, &xf.inverse(), &flipped, cache) {
        *cache = SatCache::default();
        return false;
    }
    let matrix = Mat3::from_quat(xf.q);
    manifold.normal = -(matrix * manifold.normal);
    for p in &mut manifold.points[..manifold.count] {
        p.point = matrix * p.point + xf.p;
        p.pair = p.pair.flip();
    }
    cache.kind = AXIS_FACE_B;
    cache.index_a = query.index_a as u8;
    cache.index_b = query.index_b as u8;
    true
}

/// Box3D `b3BuildEdgeContact`: one point between the closest points of two edges.
fn build_edge_contact(
    manifold: &mut LocalManifold,
    hull_a: &HullView<'_>,
    hull_b: &HullView<'_>,
    xf: &Transform,
    query: &SeparatingAxis,
    cache: &mut SatCache,
) -> bool {
    let edge_a = hull_a.edges[query.index_a as usize];
    let twin_a = hull_a.edges[edge_a.twin as usize];
    let p_a = hull_a.points[edge_a.origin as usize];
    let q_a = hull_a.points[twin_a.origin as usize];
    let edge_b = hull_b.edges[query.index_b as usize];
    let twin_b = hull_b.edges[edge_b.twin as usize];
    let p_b = xf.apply(hull_b.points[edge_b.origin as usize]);
    let q_b = xf.apply(hull_b.points[twin_b.origin as usize]);
    let result = line_distance(p_a, q_a - p_a, p_b, q_b - p_b);
    if !result.is_within_segments() {
        *cache = SatCache::default();
        return false;
    }
    let separation = query.normal.dot(result.point2 - result.point1);
    let point = (result.point1 + result.point2) * 0.5;
    *manifold = LocalManifold::single(
        query.normal,
        point,
        separation,
        FeaturePair::new(
            SHAPE_A,
            query.index_a as usize,
            SHAPE_B,
            query.index_b as usize,
        ),
    );
    *cache = SatCache {
        separation,
        kind: AXIS_EDGE_PAIR,
        index_a: query.index_a as u8,
        index_b: query.index_b as u8,
        hit: cache.hit,
    };
    true
}

/// Box3D `b3GetFaceSeparation` with `b3GetSupportWide`: the separation of the vertex of
/// `points` deepest along `-direction`, `plane_separation - max dot(direction, v)`. As in
/// Box3D the support minimizes `bias - dot` with the vertex index in the low seven mantissa
/// bits, so near ties go to the lowest index.
fn face_separation(direction: Vec3, plane_separation: f32, hull: &HullView<'_>) -> (f32, usize) {
    let bias = direction.dot(hull.aabb_center) + 1.0625 * direction.abs().dot(hull.aabb_extents);
    let mut min_value = f32::INFINITY;
    for (index, point) in hull.points.iter().enumerate() {
        let value = bias - direction.dot(*point);
        let augmented = f32::from_bits((value.to_bits() & !0x7F) | index as u32);
        min_value = min_value.min(augmented);
    }
    let vertex = (min_value.to_bits() & 0x7F) as usize;
    let support = direction.dot(hull.points[vertex]);
    (plane_separation - support, vertex)
}

/// Box3D's separating axis test (`b3ComputeSeparatingAxis`, the scalar form of its SIMD
/// kernel without the candidate culling, which never changes the answer): the faces of A
/// against B's vertices, B's faces against A's, then the edge pairs that build a face of the
/// Minkowski difference. With `early_return` it stops at the first axis separating by more
/// than the speculative distance.
fn compute_separating_axis(
    hull_a: &HullView<'_>,
    hull_b: &HullView<'_>,
    xf: &Transform,
    early_return: bool,
) -> AxisQuery {
    let r = Mat3::from_quat(xf.q);
    let inv_r = r.transpose();
    let mut result = AxisQuery {
        face_a: SeparatingAxis::NONE,
        face_b: SeparatingAxis::NONE,
        edge: SeparatingAxis::NONE,
        separated: AXIS_INVALID,
    };

    for (i, plane) in hull_a.planes.iter().enumerate() {
        let direction = -(inv_r * plane.normal);
        let plane_separation = plane.normal.dot(xf.p) - plane.offset;
        let (separation, vertex) = face_separation(direction, plane_separation, hull_b);
        if separation > result.face_a.separation {
            result.face_a = SeparatingAxis {
                normal: plane.normal,
                separation,
                index_a: i as i32,
                index_b: vertex as i32,
            };
            if separation > SPECULATIVE_DISTANCE && early_return {
                result.separated = AXIS_FACE_A;
                return result;
            }
        }
    }

    for (i, plane) in hull_b.planes.iter().enumerate() {
        let direction = -(r * plane.normal);
        let plane_separation = direction.dot(xf.p) - plane.offset;
        let (separation, vertex) = face_separation(direction, plane_separation, hull_a);
        if separation > result.face_b.separation {
            result.face_b = SeparatingAxis {
                normal: direction,
                separation,
                index_a: vertex as i32,
                index_b: i as i32,
            };
            if separation > SPECULATIVE_DISTANCE && early_return {
                result.separated = AXIS_FACE_B;
                return result;
            }
        }
    }

    // B's face normals and vertices in A's frame, negated.
    let normal_b = |face: u8| -(r * hull_b.planes[face as usize].normal);
    let vertex_b = |vertex: u8| -(r * hull_b.points[vertex as usize] + xf.p);
    let eps = -LINEAR_SLOP * LINEAR_SLOP;
    let squared_tolerance = PARALLEL_EDGE_TOL * PARALLEL_EDGE_TOL;
    for j in (0..hull_b.edges.len()).step_by(2) {
        let (edge, twin) = (hull_b.edges[j], hull_b.edges[j + 1]);
        let c = normal_b(edge.face);
        let d = normal_b(twin.face);
        let bv0 = vertex_b(edge.origin);
        let dc = vertex_b(twin.origin) - bv0;
        for i in (0..hull_a.edges.len()).step_by(2) {
            let (edge_a, twin_a) = (hull_a.edges[i], hull_a.edges[i + 1]);
            let av0 = hull_a.points[edge_a.origin as usize];
            let direction = hull_a.points[twin_a.origin as usize] - av0;
            // The arcs of the two edges cross on the Gauss map.
            let cba = c.dot(direction);
            let dba = d.dot(direction);
            if cba * dba >= eps {
                continue;
            }
            let n0 = hull_a.planes[edge_a.face as usize].normal;
            let n1 = hull_a.planes[twin_a.face as usize].normal;
            let adc = n0.dot(dc);
            let bdc = n1.dot(dc);
            if adc * bdc >= eps || cba * bdc >= eps {
                continue;
            }
            // Reject nearly parallel edges.
            let tolerance = squared_tolerance
                * (direction.x * direction.x
                    + direction.y * direction.y
                    + direction.z * direction.z);
            if (cba * cba).max(dba * dba) <= tolerance {
                continue;
            }
            let t = -cba / (dba - cba);
            let n = c + (d - c) * t;
            let n = n * (1.0 / n.dot(n).sqrt());
            let separation = -(av0 + bv0).dot(n);
            if separation > result.edge.separation {
                result.edge = SeparatingAxis {
                    normal: n,
                    separation,
                    index_a: i as i32,
                    index_b: j as i32,
                };
                if separation > SPECULATIVE_DISTANCE && early_return {
                    result.separated = AXIS_EDGE_PAIR;
                    return result;
                }
            }
        }
    }
    result
}

/// Box3D `b3CollideHulls`: tries the cached axis first, then the full separating axis test;
/// face contacts unless an edge pair is clearly better.
pub(crate) fn collide_hulls(
    hull_a: &HullView<'_>,
    hull_b: &HullView<'_>,
    xf: &Transform,
    cache: &mut SatCache,
) -> LocalManifold {
    let mut manifold = LocalManifold::default();
    cache.hit = 0;

    match cache.kind {
        AXIS_FACE_A if (cache.index_a as usize) < hull_a.planes.len() => {
            let plane = hull_a.planes[cache.index_a as usize];
            let direction_in_b = -crate::math::inv_rotate(xf.q, plane.normal);
            let vertex = hull_b.support_vertex(direction_in_b);
            let separation = plane.separation(xf.apply(hull_b.points[vertex]));
            if separation >= SPECULATIVE_DISTANCE {
                cache.hit = 1;
                return manifold;
            }
            let query = SeparatingAxis {
                normal: plane.normal,
                separation: 0.0,
                index_a: i32::from(cache.index_a),
                index_b: vertex as i32,
            };
            let mut local = SatCache::default();
            let touching =
                build_face_a_contact(&mut manifold, hull_a, hull_b, xf, &query, &mut local);
            if touching && (cache.separation - local.separation).abs() < LINEAR_SLOP {
                cache.hit = 1;
                return manifold;
            }
        }
        AXIS_FACE_B if (cache.index_b as usize) < hull_b.planes.len() => {
            let plane = hull_b.planes[cache.index_b as usize];
            let direction_in_a = -crate::math::rotate(xf.q, plane.normal);
            let vertex = hull_a.support_vertex(direction_in_a);
            let separation = plane.separation(xf.apply_inverse(hull_a.points[vertex]));
            if separation >= SPECULATIVE_DISTANCE {
                cache.hit = 1;
                return manifold;
            }
            let query = SeparatingAxis {
                normal: -plane.normal,
                separation: 0.0,
                index_a: vertex as i32,
                index_b: i32::from(cache.index_b),
            };
            let mut local = SatCache::default();
            let touching =
                build_face_b_contact(&mut manifold, hull_a, hull_b, xf, &query, &mut local);
            if touching && (cache.separation - local.separation).abs() < LINEAR_SLOP {
                cache.hit = 1;
                return manifold;
            }
        }
        AXIS_EDGE_PAIR
            if (cache.index_a as usize) + 1 < hull_a.edges.len()
                && (cache.index_b as usize) + 1 < hull_b.edges.len() =>
        {
            let index_a = cache.index_a as usize;
            let (edge1, twin1) = (hull_a.edges[index_a], hull_a.edges[index_a + 1]);
            let p_a = hull_a.points[edge1.origin as usize];
            let q_a = hull_a.points[twin1.origin as usize];
            let e_a = q_a - p_a;
            let u_a = hull_a.planes[edge1.face as usize].normal;
            let v_a = hull_a.planes[twin1.face as usize].normal;
            let index_b = cache.index_b as usize;
            let (edge2, twin2) = (hull_b.edges[index_b], hull_b.edges[index_b + 1]);
            let p_b = xf.apply(hull_b.points[edge2.origin as usize]);
            let q_b = xf.apply(hull_b.points[twin2.origin as usize]);
            let e_b = q_b - p_b;
            let u_b = crate::math::rotate(xf.q, hull_b.planes[edge2.face as usize].normal);
            let v_b = crate::math::rotate(xf.q, hull_b.planes[twin2.face as usize].normal);
            let cba = u_b.dot(e_a);
            let dba = v_b.dot(e_a);
            let adc = -u_a.dot(e_b);
            let bdc = -v_a.dot(e_b);
            let tolerance = PARALLEL_EDGE_TOL * PARALLEL_EDGE_TOL;
            if cba * dba < 0.0
                && adc * bdc < 0.0
                && cba * bdc > 0.0
                && (cba * cba).max(dba * dba) >= tolerance * e_a.length_squared()
            {
                let t = cba / (cba - dba);
                let axis = normalize(lerp(u_b, v_b, t));
                let separation = axis.dot(q_a - q_b);
                if separation > SPECULATIVE_DISTANCE {
                    cache.hit = 1;
                    return manifold;
                }
                let query = SeparatingAxis {
                    normal: -axis,
                    separation: 0.0,
                    index_a: i32::from(cache.index_a),
                    index_b: i32::from(cache.index_b),
                };
                let mut local = SatCache::default();
                let touching =
                    build_edge_contact(&mut manifold, hull_a, hull_b, xf, &query, &mut local);
                if touching && (cache.separation - local.separation).abs() < LINEAR_SLOP {
                    cache.hit = 1;
                    return manifold;
                }
            }
        }
        _ => {}
    }

    manifold = LocalManifold::default();
    *cache = SatCache::default();

    let query = compute_separating_axis(hull_a, hull_b, xf, true);
    if query.separated != AXIS_INVALID {
        let axis = match query.separated {
            AXIS_FACE_A => query.face_a,
            AXIS_FACE_B => query.face_b,
            _ => query.edge,
        };
        *cache = SatCache {
            separation: axis.separation,
            kind: query.separated,
            index_a: axis.index_a as u8,
            index_b: axis.index_b as u8,
            hit: 0,
        };
        return manifold;
    }

    if query.face_a.separation > query.face_b.separation {
        build_face_a_contact(&mut manifold, hull_a, hull_b, xf, &query.face_a, cache);
    } else {
        build_face_b_contact(&mut manifold, hull_a, hull_b, xf, &query.face_b, cache);
    }

    let edge = query.edge;
    if edge.index_a < 0 {
        // All edges are parallel.
        return manifold;
    }
    let face_separation = query.face_a.separation.max(query.face_b.separation);
    let clip_separation = cache.separation;
    // The face contact can be empty when it is not the axis of maximum separation; take the
    // edge contact then, or when it is clearly better.
    if (manifold.count == 0 && edge.separation > face_separation)
        || edge.separation > clip_separation + LINEAR_SLOP
    {
        let mut edge_manifold = LocalManifold::default();
        let mut edge_cache = SatCache::default();
        build_edge_contact(
            &mut edge_manifold,
            hull_a,
            hull_b,
            xf,
            &edge,
            &mut edge_cache,
        );
        // Speculation can miss a vertex-vertex contact; keep the face points then.
        if edge_manifold.count == 1 {
            manifold = edge_manifold;
            *cache = edge_cache;
        }
    }
    manifold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hull::BoxHull;
    use korppi_math::Quat;

    fn unit_box() -> BoxHull {
        BoxHull::new(Vec3::splat(0.5))
    }

    #[test]
    fn stacked_boxes_touch_on_four_points() {
        let (a, b) = (unit_box(), unit_box());
        let mut cache = SatCache::default();
        let xf = Transform::new(Vec3::new(0.1, 0.99, -0.2), Quat::IDENTITY);
        let m = collide_hulls(&a.view(), &b.view(), &xf, &mut cache);
        assert_eq!(m.count, 4);
        assert!((m.normal - Vec3::Y).length() < 1e-6);
        for p in m.points() {
            assert!((p.separation + 0.01).abs() < 1e-5, "{p:?}");
            assert!(
                (p.point.y - 0.495).abs() < 1e-5,
                "half-way between the faces: {p:?}"
            );
        }
        assert_ne!(cache.kind, AXIS_INVALID, "the touching axis is cached");

        // The cached face gives the same manifold.
        let again = collide_hulls(&a.view(), &b.view(), &xf, &mut cache);
        assert_eq!(cache.hit, 1);
        assert_eq!(again.count, 4);

        // 1.5 cm apart: speculative points; 3 cm apart: nothing.
        let xf = Transform::new(Vec3::new(0.0, 1.015, 0.0), Quat::IDENTITY);
        let m = collide_hulls(&a.view(), &b.view(), &xf, &mut SatCache::default());
        assert_eq!(m.count, 4);
        let xf = Transform::new(Vec3::new(0.0, 1.03, 0.0), Quat::IDENTITY);
        let m = collide_hulls(&a.view(), &b.view(), &xf, &mut SatCache::default());
        assert_eq!(m.count, 0);
    }

    #[test]
    fn spheres_and_capsules_meet_half_way() {
        let a = Sphere {
            center: Vec3::ZERO,
            radius: 0.5,
        };
        let xf = Transform::new(Vec3::new(0.9, 0.0, 0.0), Quat::IDENTITY);
        let m = collide_spheres(&a, &a, &xf);
        assert_eq!(m.count, 1);
        assert!((m.points[0].separation + 0.1).abs() < 1e-6);
        assert!((m.points[0].point - Vec3::new(0.45, 0.0, 0.0)).length() < 1e-6);

        let capsule = Capsule {
            center1: Vec3::new(0.0, -0.5, 0.0),
            center2: Vec3::new(0.0, 0.5, 0.0),
            radius: 0.25,
        };
        // Parallel capsules side by side, 1 cm deep: two points.
        let xf = Transform::new(Vec3::new(0.49, 0.0, 0.0), Quat::IDENTITY);
        let m = collide_capsules(&capsule, &capsule, &xf);
        assert_eq!(m.count, 2);
        assert!((m.normal - Vec3::X).length() < 1e-6);

        // A capsule lying on a box face, 1 cm above it: two speculative points.
        let hull = BoxHull::new(Vec3::splat(0.5));
        let xf = Transform::new(
            Vec3::new(0.0, 0.76, 0.0),
            Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        );
        let m = collide_hull_and_capsule(&hull.view(), &capsule, &xf, &mut SimplexCache::default());
        assert_eq!(m.count, 2);
        for p in m.points() {
            assert!((p.separation - 0.01).abs() < 1e-5, "{p:?}");
        }
    }
}
