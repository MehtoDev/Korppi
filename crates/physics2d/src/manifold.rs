//! Contact manifolds of shape pairs, ported from Box2D v3's `manifold.c`.
//!
//! Each function computes a [`LocalManifold2d`] in shape A's frame, with shape B placed by the
//! relative transform `xf` (B in A). A manifold has up to two points; a point's position lies
//! midway between the two surfaces, and its separation is negative when the shapes overlap.
//! Shapes further apart than the speculative distance, four times `linear_slop` (Box2D
//! [`SPECULATIVE_DISTANCE`] in metres), give no points. Capsules and segments
//! collide with polygons as two-vertex rounded polygons (Box2D `b2MakeCapsule`). Chain segments
//! collide on their right side only, and against capsules and polygons warm start their
//! distance query from the contact's [`SimplexCache2d`].

use crate::distance::{
    segment_distance, shape_distance, ShapeProxy2d, SimplexCache2d, LINEAR_SLOP,
};
use crate::math::{cross, left_perp, length_and_normalize, lerp, normalize, right_perp};
use crate::shape::{
    Capsule2d, ChainSegment2d, Circle2d, Polygon2d, Segment2d, MAX_POLYGON_VERTICES,
};
use korppi_math::{Transform2d, Vec2};

/// Distance below which contact points are kept, so the solver can prevent penetration before
/// it happens (Box2D `B2_SPECULATIVE_DISTANCE`, four linear slops).
pub const SPECULATIVE_DISTANCE: f32 = 4.0 * LINEAR_SLOP;

/// A contact point in shape A's frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LocalManifoldPoint2d {
    /// Contact point, midway between the surfaces.
    pub point: Vec2,
    /// Separation along the normal; negative when overlapping.
    pub separation: f32,
    /// Identifies the pair of features that made the point, for warm starting.
    pub id: u16,
}

/// A contact manifold in shape A's frame (Box2D `b2LocalManifold`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LocalManifold2d {
    /// Unit normal pointing from shape A to shape B.
    pub normal: Vec2,
    /// The points; only the first `point_count` are used.
    pub points: [LocalManifoldPoint2d; 2],
    /// Number of points, 0 to 2.
    pub point_count: usize,
}

impl LocalManifold2d {
    fn single(normal: Vec2, point: Vec2, separation: f32, id: u16) -> Self {
        let mut manifold = Self {
            normal,
            point_count: 1,
            ..Self::default()
        };
        manifold.points[0] = LocalManifoldPoint2d {
            point,
            separation,
            id,
        };
        manifold
    }
}

/// Feature id of a point made by feature `a` of one shape and feature `b` of the other.
const fn make_id(a: usize, b: usize) -> u16 {
    ((a as u16 & 0xFF) << 8) | (b as u16 & 0xFF)
}

/// Circle A and circle B (Box2D `b2CollideCircles`).
#[must_use]
pub fn collide_circles(
    a: &Circle2d,
    b: &Circle2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    let speculative_distance = 4.0 * linear_slop;
    let point_a = a.center;
    let point_b = xf.transform_point(b.center);
    let distance = (point_b - point_a).length();
    let normal = (point_b - point_a).normalize_or_zero();
    let separation = distance - a.radius - b.radius;
    if separation > speculative_distance {
        return LocalManifold2d::default();
    }
    let c_a = point_a + normal * a.radius;
    let c_b = point_b - normal * b.radius;
    LocalManifold2d::single(normal, (c_a + c_b) * 0.5, separation, 0)
}

/// Capsule A and circle B (Box2D `b2CollideCapsuleAndCircle`).
#[must_use]
pub fn collide_capsule_and_circle(
    a: &Capsule2d,
    b: &Circle2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    let speculative_distance = 4.0 * linear_slop;
    let p_b = xf.transform_point(b.center);
    let (p1, p2) = (a.point1, a.point2);
    let e = p2 - p1;
    let s1 = (p_b - p1).dot(e);
    let s2 = (p2 - p_b).dot(e);
    let p_a = if s1 < 0.0 {
        p1
    } else if s2 < 0.0 {
        p2
    } else {
        p1 + e * (s1 / e.dot(e))
    };
    let distance = (p_b - p_a).length();
    let normal = (p_b - p_a).normalize_or_zero();
    let separation = distance - a.radius - b.radius;
    if separation > speculative_distance {
        return LocalManifold2d::default();
    }
    let c_a = p_a + normal * a.radius;
    let c_b = p_b - normal * b.radius;
    LocalManifold2d::single(normal, (c_a + c_b) * 0.5, separation, 0)
}

/// Polygon A and circle B (Box2D `b2CollidePolygonAndCircle`).
#[must_use]
pub fn collide_polygon_and_circle(
    a: &Polygon2d,
    b: &Circle2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    let speculative_distance = 4.0 * linear_slop;
    let poly = a;
    let center = xf.transform_point(b.center);
    let (radius_a, radius_b) = (poly.radius, b.radius);
    let radius = radius_a + radius_b;

    // The edge the circle centre is furthest outside.
    let mut normal_index = 0;
    let mut separation = f32::MIN;
    for i in 0..poly.count {
        let s = poly.normals[i].dot(center - poly.vertices[i]);
        if s > separation {
            separation = s;
            normal_index = i;
        }
    }
    if separation > radius + speculative_distance {
        return LocalManifold2d::default();
    }

    let v1 = poly.vertices[normal_index];
    let v2 = poly.vertices[poly.next(normal_index)];
    let u1 = (center - v1).dot(v2 - v1);
    let u2 = (center - v2).dot(v1 - v2);
    let vertex_contact = |v: Vec2| {
        let normal = (center - v).normalize_or_zero();
        if (center - v).dot(normal) > radius + speculative_distance {
            return LocalManifold2d::default();
        }
        let c_a = v + normal * radius_a;
        let c_b = center - normal * radius_b;
        LocalManifold2d::single(normal, (c_a + c_b) * 0.5, (c_b - c_a).dot(normal), 0)
    };
    if u1 < 0.0 && separation > f32::EPSILON {
        // Closest to v1, safely outside the polygon.
        vertex_contact(v1)
    } else if u2 < 0.0 && separation > f32::EPSILON {
        vertex_contact(v2)
    } else {
        // Between v1 and v2; the centre may be inside the polygon.
        let normal = poly.normals[normal_index];
        let c_a = center + normal * (radius_a - (center - v1).dot(normal));
        let c_b = center - normal * radius_b;
        LocalManifold2d::single(normal, (c_a + c_b) * 0.5, separation - radius, 0)
    }
}

/// Capsule A and capsule B (Box2D `b2CollideCapsules`): closest points of the core segments,
/// clipped to two points when the segments overlap along each other.
#[must_use]
pub fn collide_capsules(
    a: &Capsule2d,
    b: &Capsule2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    let speculative_distance = 4.0 * linear_slop;
    // Work relative to A's first point for round-off.
    let origin = a.point1;
    let shifted = Transform2d::from_translation_rotation(xf.translation - origin, xf.rotation);
    let p1 = Vec2::ZERO;
    let q1 = a.point2 - origin;
    let p2 = shifted.transform_point(b.point1);
    let q2 = shifted.transform_point(b.point2);

    let closest = segment_distance(p1, q1, p2, q2);
    let (radius_a, radius_b) = (a.radius, b.radius);
    let radius = radius_a + radius_b;
    let max_distance = radius + speculative_distance;
    if closest.distance_squared > max_distance * max_distance {
        return LocalManifold2d::default();
    }
    let distance = closest.distance_squared.sqrt();

    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let (length1, length2) = (d1.length(), d2.length());
    let u1 = d1.normalize_or_zero();
    let u2 = d2.normalize_or_zero();

    // Does segment B project outside segment A, or A outside B?
    let fp2 = (p2 - p1).dot(u1);
    let fq2 = (q2 - p1).dot(u1);
    let outside_a = (fp2 <= 0.0 && fq2 <= 0.0) || (fp2 >= length1 && fq2 >= length1);
    let fp1 = (p1 - p2).dot(u2);
    let fq1 = (q1 - p2).dot(u2);
    let outside_b = (fp1 <= 0.0 && fq1 <= 0.0) || (fp1 >= length2 && fq1 >= length2);

    let mut manifold = LocalManifold2d::default();
    if !outside_a && !outside_b {
        // Reference edge by separating axis; may give points too far apart, in which case
        // the single closest point is used instead.
        let edge_separation = |u: Vec2, from: Vec2, p: Vec2, q: Vec2| {
            let normal = Vec2::new(-u.y, u.x);
            let s1 = (p - from).dot(normal);
            let s2 = (q - from).dot(normal);
            let positive = s1.min(s2);
            let negative = (-s1).min(-s2);
            if positive > negative {
                (normal, positive)
            } else {
                (-normal, negative)
            }
        };
        let (normal_a, separation_a) = edge_separation(u1, p1, p2, q2);
        let (normal_b, separation_b) = edge_separation(u2, p2, p1, q1);

        // Biased toward A to avoid feature flip-flop.
        if separation_a + 0.1 * linear_slop >= separation_b {
            let (mut cp, mut cq) = (p2, q2);
            if fp2 < 0.0 && fq2 > 0.0 {
                cp = p2.lerp(q2, (0.0 - fp2) / (fq2 - fp2));
            } else if fq2 < 0.0 && fp2 > 0.0 {
                cq = q2.lerp(p2, (0.0 - fq2) / (fp2 - fq2));
            }
            if fp2 > length1 && fq2 < length1 {
                cp = p2.lerp(q2, (fp2 - length1) / (fp2 - fq2));
            } else if fq2 > length1 && fp2 < length1 {
                cq = q2.lerp(p2, (fq2 - length1) / (fq2 - fp2));
            }
            let sp = (cp - p1).dot(normal_a);
            let sq = (cq - p1).dot(normal_a);
            if sp <= distance + linear_slop || sq <= distance + linear_slop {
                manifold.normal = normal_a;
                manifold.points[0] = LocalManifoldPoint2d {
                    point: cp + normal_a * (0.5 * (radius_a - radius_b - sp)),
                    separation: sp - radius,
                    id: make_id(0, 0),
                };
                manifold.points[1] = LocalManifoldPoint2d {
                    point: cq + normal_a * (0.5 * (radius_a - radius_b - sq)),
                    separation: sq - radius,
                    id: make_id(0, 1),
                };
                manifold.point_count = 2;
            }
        } else {
            let (mut cp, mut cq) = (p1, q1);
            if fp1 < 0.0 && fq1 > 0.0 {
                cp = p1.lerp(q1, (0.0 - fp1) / (fq1 - fp1));
            } else if fq1 < 0.0 && fp1 > 0.0 {
                cq = q1.lerp(p1, (0.0 - fq1) / (fp1 - fq1));
            }
            if fp1 > length2 && fq1 < length2 {
                cp = p1.lerp(q1, (fp1 - length2) / (fp1 - fq1));
            } else if fq1 > length2 && fp1 < length2 {
                cq = q1.lerp(p1, (fq1 - length2) / (fq1 - fp1));
            }
            let sp = (cp - p2).dot(normal_b);
            let sq = (cq - p2).dot(normal_b);
            if sp <= distance + linear_slop || sq <= distance + linear_slop {
                // The normal always points from A to B.
                manifold.normal = -normal_b;
                manifold.points[0] = LocalManifoldPoint2d {
                    point: cp + normal_b * (0.5 * (radius_b - radius_a - sp)),
                    separation: sp - radius,
                    id: make_id(0, 0),
                };
                manifold.points[1] = LocalManifoldPoint2d {
                    point: cq + normal_b * (0.5 * (radius_b - radius_a - sq)),
                    separation: sq - radius,
                    id: make_id(1, 0),
                };
                manifold.point_count = 2;
            }
        }
    }

    if manifold.point_count == 0 {
        // Single point at the closest points.
        let offset = closest.closest2 - closest.closest1;
        let normal = if offset.length_squared() > f32::EPSILON * f32::EPSILON {
            offset.normalize()
        } else {
            Vec2::new(-u1.y, u1.x)
        };
        let c1 = closest.closest1 + normal * radius_a;
        let c2 = closest.closest2 - normal * radius_b;
        let i1 = usize::from(closest.fraction1 != 0.0);
        let i2 = usize::from(closest.fraction2 != 0.0);
        manifold =
            LocalManifold2d::single(normal, (c1 + c2) * 0.5, distance - radius, make_id(i1, i2));
    }

    for point in &mut manifold.points[..manifold.point_count] {
        point.point += origin;
    }
    manifold
}

/// Segment A and capsule B (Box2D `b2CollideSegmentAndCapsule`).
#[must_use]
pub fn collide_segment_and_capsule(
    a: &Segment2d,
    b: &Capsule2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    collide_capsules(&Capsule2d::new(a.point1, a.point2, 0.0), b, xf, linear_slop)
}

/// Segment A and circle B (Box2D `b2CollideSegmentAndCircle`).
#[must_use]
pub fn collide_segment_and_circle(
    a: &Segment2d,
    b: &Circle2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    collide_capsule_and_circle(&Capsule2d::new(a.point1, a.point2, 0.0), b, xf, linear_slop)
}

/// Polygon A and capsule B (Box2D `b2CollidePolygonAndCapsule`).
#[must_use]
pub fn collide_polygon_and_capsule(
    a: &Polygon2d,
    b: &Capsule2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    collide_local_polygons(
        a,
        &Polygon2d::capsule(b.point1, b.point2, b.radius),
        xf,
        linear_slop,
    )
}

/// Segment A and polygon B (Box2D `b2CollideSegmentAndPolygon`).
#[must_use]
pub fn collide_segment_and_polygon(
    a: &Segment2d,
    b: &Polygon2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    collide_local_polygons(
        &Polygon2d::capsule(a.point1, a.point2, 0.0),
        b,
        xf,
        linear_slop,
    )
}

/// Polygon A and polygon B (Box2D `b2CollidePolygons`).
#[must_use]
pub fn collide_polygons(
    a: &Polygon2d,
    b: &Polygon2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    collide_local_polygons(a, b, xf, linear_slop)
}

/// The edge of `poly1` along which `poly2` is furthest outside, and that separation.
fn find_max_separation(poly1: &Polygon2d, poly2: &Polygon2d) -> (usize, f32) {
    let mut best_index = 0;
    let mut max_separation = f32::MIN;
    for i in 0..poly1.count {
        let (n, v1) = (poly1.normals[i], poly1.vertices[i]);
        let si = poly2.vertices[..poly2.count]
            .iter()
            .map(|&v2| n.dot(v2 - v1))
            .fold(f32::MAX, f32::min);
        if si > max_separation {
            max_separation = si;
            best_index = i;
        }
    }
    (best_index, max_separation)
}

/// Clips the incident edge against the reference edge's side planes (Box2D `b2ClipPolygons`).
/// The reference edge is A's `edge_a`, or B's `edge_b` when `flip` is set.
fn clip_polygons(
    poly_a: &Polygon2d,
    poly_b: &Polygon2d,
    edge_a: usize,
    edge_b: usize,
    flip: bool,
) -> LocalManifold2d {
    let (poly1, i11, poly2, i21) = if flip {
        (poly_b, edge_b, poly_a, edge_a)
    } else {
        (poly_a, edge_a, poly_b, edge_b)
    };
    let (i12, i22) = (poly1.next(i11), poly2.next(i21));
    let normal = poly1.normals[i11];
    let (v11, v12) = (poly1.vertices[i11], poly1.vertices[i12]);
    let (v21, v22) = (poly2.vertices[i21], poly2.vertices[i22]);
    let tangent = Vec2::new(-normal.y, normal.x);

    let lower1 = 0.0;
    let upper1 = (v12 - v11).dot(tangent);
    // The incident edge runs the opposite way.
    let upper2 = (v21 - v11).dot(tangent);
    let lower2 = (v22 - v11).dot(tangent);
    if upper2 < lower1 || upper1 < lower2 {
        return LocalManifold2d::default();
    }
    let v_lower = if lower2 < lower1 && upper2 - lower2 > f32::EPSILON {
        v22.lerp(v21, (lower1 - lower2) / (upper2 - lower2))
    } else {
        v22
    };
    let v_upper = if upper2 > upper1 && upper2 - lower2 > f32::EPSILON {
        v22.lerp(v21, (upper1 - lower2) / (upper2 - lower2))
    } else {
        v21
    };
    let separation_lower = (v_lower - v11).dot(normal);
    let separation_upper = (v_upper - v11).dot(normal);
    let (r1, r2) = (poly1.radius, poly2.radius);
    // Put the points midway between the rounded surfaces.
    let v_lower = v_lower + normal * (0.5 * (r1 - r2 - separation_lower));
    let v_upper = v_upper + normal * (0.5 * (r1 - r2 - separation_upper));
    let radius = r1 + r2;

    let mut manifold = LocalManifold2d {
        point_count: 2,
        ..LocalManifold2d::default()
    };
    if flip {
        manifold.normal = -normal;
        manifold.points[0] = LocalManifoldPoint2d {
            point: v_upper,
            separation: separation_upper - radius,
            id: make_id(i21, i12),
        };
        manifold.points[1] = LocalManifoldPoint2d {
            point: v_lower,
            separation: separation_lower - radius,
            id: make_id(i22, i11),
        };
    } else {
        manifold.normal = normal;
        manifold.points[0] = LocalManifoldPoint2d {
            point: v_lower,
            separation: separation_lower - radius,
            id: make_id(i11, i22),
        };
        manifold.points[1] = LocalManifoldPoint2d {
            point: v_upper,
            separation: separation_upper - radius,
            id: make_id(i12, i21),
        };
    }
    manifold
}

fn collide_local_polygons(
    a: &Polygon2d,
    b: &Polygon2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    let speculative_distance = 4.0 * linear_slop;
    // Work relative to A's first vertex for round-off.
    let origin = a.vertices[0];
    let shifted = Transform2d::from_translation_rotation(xf.translation - origin, xf.rotation);
    let mut poly_a = *a;
    for v in &mut poly_a.vertices[..poly_a.count] {
        *v -= origin;
    }
    let mut poly_b = *b;
    for i in 0..poly_b.count {
        poly_b.vertices[i] = shifted.transform_point(b.vertices[i]);
        poly_b.normals[i] = shifted.rotation.rotate(b.normals[i]);
    }
    let radius = poly_a.radius + poly_b.radius;

    let (mut edge_a, separation_a) = find_max_separation(&poly_a, &poly_b);
    if separation_a > speculative_distance + radius {
        return LocalManifold2d::default();
    }
    let (mut edge_b, separation_b) = find_max_separation(&poly_b, &poly_a);
    if separation_b > speculative_distance + radius {
        return LocalManifold2d::default();
    }

    // The incident edge is the one most anti-parallel to the reference normal.
    let most_anti_parallel = |direction: Vec2, poly: &Polygon2d| {
        let mut index = 0;
        let mut min_dot = f32::MAX;
        for i in 0..poly.count {
            let dot = direction.dot(poly.normals[i]);
            if dot < min_dot {
                min_dot = dot;
                index = i;
            }
        }
        index
    };
    let flip = separation_a < separation_b;
    if flip {
        edge_a = most_anti_parallel(poly_b.normals[edge_b], &poly_a);
    } else {
        edge_b = most_anti_parallel(poly_a.normals[edge_a], &poly_b);
    }

    let mut manifold;
    if separation_a > 0.1 * linear_slop || separation_b > 0.1 * linear_slop {
        // Separated (or rounded): the closest features may be two vertices, which face
        // clipping would miss.
        let (i11, i12) = (edge_a, poly_a.next(edge_a));
        let (i21, i22) = (edge_b, poly_b.next(edge_b));
        let (v11, v12) = (poly_a.vertices[i11], poly_a.vertices[i12]);
        let (v21, v22) = (poly_b.vertices[i21], poly_b.vertices[i22]);
        let closest = segment_distance(v11, v12, v21, v22);
        let distance = closest.distance_squared.sqrt();
        let separation = distance - radius;
        if separation > speculative_distance {
            return LocalManifold2d::default();
        }
        manifold = clip_polygons(&poly_a, &poly_b, edge_a, edge_b, flip);
        let min_separation = manifold.points[..manifold.point_count]
            .iter()
            .map(|p| p.separation)
            .fold(f32::MAX, f32::min);
        if separation + 0.1 * linear_slop < min_separation {
            let vertex_pair = match (closest.fraction1, closest.fraction2) {
                (f1, f2) if f1 == 0.0 && f2 == 0.0 => Some((v11, v21, i11, i21)),
                (f1, f2) if f1 == 0.0 && f2 == 1.0 => Some((v11, v22, i11, i22)),
                (f1, f2) if f1 == 1.0 && f2 == 0.0 => Some((v12, v21, i12, i21)),
                (f1, f2) if f1 == 1.0 && f2 == 1.0 => Some((v12, v22, i12, i22)),
                _ => None,
            };
            if let Some((va, vb, ia, ib)) = vertex_pair {
                let normal = (vb - va) / distance;
                let c1 = va + normal * poly_a.radius;
                let c2 = vb - normal * poly_b.radius;
                manifold = LocalManifold2d::single(
                    normal,
                    (c1 + c2) * 0.5,
                    distance - radius,
                    make_id(ia, ib),
                );
            }
        }
    } else {
        manifold = clip_polygons(&poly_a, &poly_b, edge_a, edge_b, flip);
    }

    for point in &mut manifold.points[..manifold.point_count] {
        point.point += origin;
    }
    manifold
}

/// Chain segment A and circle B (Box2D `b2CollideChainSegmentAndCircle`): one-sided, on the
/// segment's right, and only where neither neighbouring segment (through the ghost vertices)
/// should collide instead.
#[must_use]
pub fn collide_chain_segment_and_circle(
    a: &ChainSegment2d,
    b: &Circle2d,
    xf: &Transform2d,
    linear_slop: f32,
) -> LocalManifold2d {
    let speculative_distance = 4.0 * linear_slop;
    let p_b = xf.transform_point(b.center);
    let (p1, p2) = (a.segment.point1, a.segment.point2);
    let e = p2 - p1;
    // The normal points to the right.
    if right_perp(e).dot(p_b - p1) < 0.0 {
        return LocalManifold2d::default();
    }
    // Barycentric coordinates.
    let u = e.dot(p2 - p_b);
    let v = e.dot(p_b - p1);
    let p_a = if v <= 0.0 {
        // Behind point1: in the previous segment's region?
        if (p1 - a.ghost1).dot(p_b - p1) <= 0.0 {
            return LocalManifold2d::default();
        }
        p1
    } else if u <= 0.0 {
        // Ahead of point2: in the next segment's region?
        if (a.ghost2 - p2).dot(p_b - p2) > 0.0 {
            return LocalManifold2d::default();
        }
        p2
    } else {
        let ee = e.dot(e);
        let p = Vec2::new(u * p1.x + v * p2.x, u * p1.y + v * p2.y);
        if ee > 0.0 {
            p * (1.0 / ee)
        } else {
            p1
        }
    };
    let (distance, normal) = length_and_normalize(p_b - p_a);
    let separation = distance - b.radius;
    if separation > speculative_distance {
        return LocalManifold2d::default();
    }
    let c_b = p_b + normal * -b.radius;
    LocalManifold2d::single(normal, lerp(p_a, c_b, 0.5), separation, 0)
}

/// Chain segment A and capsule B (Box2D `b2CollideChainSegmentAndCapsule`): the capsule as a
/// rounded two-vertex polygon.
#[must_use]
pub fn collide_chain_segment_and_capsule(
    a: &ChainSegment2d,
    b: &Capsule2d,
    xf: &Transform2d,
    cache: &mut SimplexCache2d,
    linear_slop: f32,
) -> LocalManifold2d {
    let polygon = Polygon2d::capsule(b.point1, b.point2, b.radius);
    collide_chain_segment_and_polygon(a, &polygon, xf, cache, linear_slop)
}

/// Box2D `b2ClipSegments`: clips segment B against segment A's extent along the tangent of
/// `normal`, with the contact points midway between the surfaces.
fn clip_segments(
    (a1, a2): (Vec2, Vec2),
    (b1, b2): (Vec2, Vec2),
    normal: Vec2,
    ra: f32,
    rb: f32,
    id1: u16,
    id2: u16,
) -> LocalManifold2d {
    let tangent = left_perp(normal);
    // Barycentric coordinates of each point relative to a1 along the tangent.
    let lower1 = 0.0;
    let upper1 = (a2 - a1).dot(tangent);
    // The incident edge runs opposite the tangent, by counter-clockwise winding.
    let upper2 = (b1 - a1).dot(tangent);
    let lower2 = (b2 - a1).dot(tangent);
    if upper2 < lower1 || upper1 < lower2 {
        return LocalManifold2d::default();
    }
    let v_lower = if lower2 < lower1 && upper2 - lower2 > f32::EPSILON {
        lerp(b2, b1, (lower1 - lower2) / (upper2 - lower2))
    } else {
        b2
    };
    let v_upper = if upper2 > upper1 && upper2 - lower2 > f32::EPSILON {
        lerp(b2, b1, (upper1 - lower2) / (upper2 - lower2))
    } else {
        b1
    };
    let separation_lower = (v_lower - a1).dot(normal);
    let separation_upper = (v_upper - a1).dot(normal);
    let radius = ra + rb;
    let mut manifold = LocalManifold2d {
        normal,
        point_count: 2,
        ..LocalManifold2d::default()
    };
    manifold.points[0] = LocalManifoldPoint2d {
        point: v_lower + normal * (0.5 * (ra - rb - separation_lower)),
        separation: separation_lower - radius,
        id: id1,
    };
    manifold.points[1] = LocalManifoldPoint2d {
        point: v_upper + normal * (0.5 * (ra - rb - separation_upper)),
        separation: separation_upper - radius,
        id: id2,
    };
    manifold
}

/// How a normal sits relative to a chain segment's neighbours (Box2D `b2NormalType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalType {
    /// Non-smooth relative to a convex vertex: skip.
    Skip,
    /// Smooth relative to a convex vertex: collide along it.
    Admit,
    /// In the region of a concave vertex: snap to the segment normal.
    Snap,
}

/// A chain segment's direction and its neighbours' normals (Box2D `b2ChainSegmentParams`).
#[derive(Debug, Clone, Copy)]
struct ChainSegmentParams {
    edge1: Vec2,
    normal0: Vec2,
    normal2: Vec2,
    convex1: bool,
    convex2: bool,
}

/// Box2D `b2ClassifyNormal`, on the Gauss map (box2d.org/posts/2020/06/ghost-collisions).
fn classify_normal(params: &ChainSegmentParams, normal: Vec2) -> NormalType {
    const SIN_TOL: f32 = 0.01;
    if normal.dot(params.edge1) <= 0.0 {
        // The normal points toward the segment's tail.
        if !params.convex1 {
            NormalType::Snap
        } else if cross(normal, params.normal0) > SIN_TOL {
            NormalType::Skip
        } else {
            NormalType::Admit
        }
    } else if !params.convex2 {
        NormalType::Snap
    } else if cross(params.normal2, normal) > SIN_TOL {
        NormalType::Skip
    } else {
        NormalType::Admit
    }
}

/// Chain segment A and polygon B (Box2D `b2CollideChainSegmentAndPolygon`).
#[must_use]
pub fn collide_chain_segment_and_polygon(
    a: &ChainSegment2d,
    b: &Polygon2d,
    xf: &Transform2d,
    cache: &mut SimplexCache2d,
    linear_slop: f32,
) -> LocalManifold2d {
    let none = LocalManifold2d::default();
    let speculative_distance = 4.0 * linear_slop;
    let centroid_b = xf.transform_point(b.centroid);
    let radius_b = b.radius;
    let (p1, p2) = (a.segment.point1, a.segment.point2);
    let edge1 = normalize(p2 - p1);

    const CONVEX_TOL: f32 = 0.01;
    let edge0 = normalize(p1 - a.ghost1);
    let edge2 = normalize(a.ghost2 - p2);
    let params = ChainSegmentParams {
        edge1,
        normal0: right_perp(edge0),
        normal2: right_perp(edge2),
        convex1: cross(edge0, edge1) >= CONVEX_TOL,
        convex2: cross(edge1, edge2) >= CONVEX_TOL,
    };

    // The normal points to the right.
    let normal1 = right_perp(edge1);
    let behind1 = normal1.dot(centroid_b - p1) < 0.0;
    let behind0 = !params.convex1 || params.normal0.dot(centroid_b - p1) < 0.0;
    let behind2 = !params.convex2 || params.normal2.dot(centroid_b - p2) < 0.0;
    if behind1 && behind0 && behind2 {
        // One-sided collision.
        return none;
    }

    // Polygon B in A's frame.
    let count = b.count;
    let mut vertices = [Vec2::ZERO; MAX_POLYGON_VERTICES];
    let mut normals = [Vec2::ZERO; MAX_POLYGON_VERTICES];
    for i in 0..count {
        vertices[i] = xf.transform_point(b.vertices[i]);
        normals[i] = xf.rotation.rotate(b.normals[i]);
    }

    // Distance does not work correctly with partial polygons.
    let output = shape_distance(
        &ShapeProxy2d::new(&[p1, p2], 0.0),
        &ShapeProxy2d::new(&vertices[..count], 0.0),
        &Transform2d::IDENTITY,
        cache,
        false,
    );
    if output.distance > radius_b + speculative_distance {
        return none;
    }

    // Snap concave normals for the partial polygon.
    let n0 = if params.convex1 {
        params.normal0
    } else {
        normal1
    };
    let n2 = if params.convex2 {
        params.normal2
    } else {
        normal1
    };
    let next = |i: usize| if i < count - 1 { i + 1 } else { 0 };

    // Index of the incident vertex, or of the incident normal, on the polygon.
    let mut incident_index = None;
    let mut incident_normal = None;

    if !behind1 && output.distance > 0.1 * linear_slop {
        // The closest features may be two vertices or an edge and a vertex even when there
        // should be face contact.
        if cache.count() == 1 {
            // Vertex-vertex collision.
            let (p_a, p_b) = (output.point_a, output.point_b);
            let normal = normalize(p_b - p_a);
            match classify_normal(&params, normal) {
                NormalType::Skip => return none,
                NormalType::Admit => {
                    return LocalManifold2d::single(
                        normal,
                        p_a,
                        output.distance - radius_b,
                        make_id(cache.index_a(0), cache.index_b(0)),
                    );
                }
                NormalType::Snap => incident_index = Some(cache.index_b(0)),
            }
        } else {
            // Vertex-edge collision.
            let (ia1, ia2) = (cache.index_a(0), cache.index_a(1));
            let (ib1, ib2) = (cache.index_b(0), cache.index_b(1));
            if ia1 == ia2 {
                // One point on A, two on B: the polygon normal most aligned with the vector
                // between the closest points.
                let to_a = output.point_a - output.point_b;
                let ib = if to_a.dot(normals[ib1]) > to_a.dot(normals[ib2]) {
                    ib1
                } else {
                    ib2
                };
                let normal_b = normals[ib];
                match classify_normal(&params, -normal_b) {
                    NormalType::Skip => return none,
                    NormalType::Admit => {
                        // The polygon edge of that normal.
                        let (ib1, ib2) = (ib, next(ib));
                        let (b1, b2) = (vertices[ib1], vertices[ib2]);
                        // The incident segment vertex.
                        let dot1 = normal_b.dot(p1 - b1);
                        let dot2 = normal_b.dot(p2 - b1);
                        let neighbour = if dot1 < dot2 { n0 } else { n2 };
                        if neighbour.dot(normal_b) < normal1.dot(normal_b) {
                            // The neighbour is incident.
                            return none;
                        }
                        let mut manifold = clip_segments(
                            (b1, b2),
                            (p1, p2),
                            normal_b,
                            radius_b,
                            0.0,
                            make_id(ib1, 1),
                            make_id(ib2, 0),
                        );
                        if manifold.point_count == 2 {
                            manifold.normal = -normal_b;
                        }
                        return manifold;
                    }
                    NormalType::Snap => incident_normal = Some(ib),
                }
            } else {
                // The incident polygon vertex.
                let dot1 = normal1.dot(vertices[ib1] - p1);
                let dot2 = normal1.dot(vertices[ib2] - p2);
                incident_index = Some(if dot1 < dot2 { ib1 } else { ib2 });
            }
        }
    } else {
        // SAT along the segment normal.
        let mut edge_separation = f32::MAX;
        for (i, &vertex) in vertices[..count].iter().enumerate() {
            let s = normal1.dot(vertex - p1);
            if s < edge_separation {
                edge_separation = s;
                incident_index = Some(i);
            }
        }
        // A convex neighbour may own the edge separation.
        for (convex, normal, point) in [
            (params.convex1, params.normal0, p1),
            (params.convex2, params.normal2, p2),
        ] {
            if !convex {
                continue;
            }
            let s = vertices[..count]
                .iter()
                .map(|&vertex| normal.dot(vertex - point))
                .fold(f32::MAX, f32::min);
            if s > edge_separation {
                edge_separation = s;
                incident_index = None;
            }
        }

        // SAT along the polygon normals.
        let mut polygon_separation = -f32::MAX;
        let mut reference_index = None;
        for i in 0..count {
            let n = normals[i];
            if classify_normal(&params, -n) != NormalType::Admit {
                continue;
            }
            let p = vertices[i];
            let s = n.dot(p2 - p).min(n.dot(p1 - p));
            if s > polygon_separation {
                polygon_separation = s;
                reference_index = Some(i);
            }
        }

        if let (true, Some(ia1)) = (polygon_separation > edge_separation, reference_index) {
            let ia2 = next(ia1);
            let (a1, a2) = (vertices[ia1], vertices[ia2]);
            let n = normals[ia1];
            let dot1 = n.dot(p1 - a1);
            let dot2 = n.dot(p2 - a1);
            let neighbour = if dot1 < dot2 { n0 } else { n2 };
            if neighbour.dot(n) < normal1.dot(n) {
                // The neighbour is incident.
                return none;
            }
            let mut manifold = clip_segments(
                (a1, a2),
                (p1, p2),
                n,
                radius_b,
                0.0,
                make_id(ia1, 1),
                make_id(ia2, 0),
            );
            if manifold.point_count == 2 {
                manifold.normal = -n;
            }
            return manifold;
        }

        if incident_index.is_none() {
            // A neighbouring segment is the separating axis.
            return none;
        }
    }

    // The segment normal: the polygon edge next to the deepest vertex most anti-parallel to it.
    let (ib1, ib2) = if let Some(ib1) = incident_normal {
        (ib1, next(ib1))
    } else {
        let Some(i2) = incident_index else {
            return none;
        };
        let i1 = if i2 > 0 { i2 - 1 } else { count - 1 };
        if normal1.dot(normals[i1]) < normal1.dot(normals[i2]) {
            (i1, i2)
        } else {
            (i2, next(i2))
        }
    };
    clip_segments(
        (p1, p2),
        (vertices[ib1], vertices[ib2]),
        normal1,
        0.0,
        radius_b,
        make_id(0, ib2),
        make_id(1, ib1),
    )
}
