//! Box3D's triangle manifold builders (`triangle_manifold.c`).
//!
//! Each builder collides one mesh triangle, given in the other shape's frame, with a sphere, a
//! capsule or a hull, from the triangle's front only, and writes up to `points.len()` points
//! midway between the surfaces, kept up to the speculative distance apart. Besides the normal
//! (from the triangle to the shape) it reports which triangle feature made the contact, its
//! face, an edge or a corner, so that the mesh contact can drop the edge and corner contacts
//! that a neighbouring face already makes. Hulls use the separating axis test with the
//! contact's cached axis; a hull seen from behind is remembered so it is not collided again
//! until it moves a linear slop.

use crate::contact_solver::{normalize, LINEAR_SLOP, SPECULATIVE_DISTANCE};
use crate::distance::{line_distance, shape_distance, ShapeProxy, SimplexCache};
use crate::hull::{HullView, Plane};
use crate::manifold::{
    clip_polygon, find_incident_face, Capsule, ClipVertex, FeaturePair, LocalPoint, SatCache,
    Sphere, AXIS_BACKSIDE, AXIS_EDGE_PAIR, AXIS_FACE_A, AXIS_FACE_B, AXIS_INVALID, MAX_CLIP_POINTS,
    SHAPE_A, SHAPE_B,
};
use crate::math::{lerp, Transform};
use korppi_math::Vec3;

/// Box3D's tolerance for nearly parallel edges in the triangle builders, squared.
const SQUARED_TOLERANCE: f32 = 0.005 * 0.005;

/// What made a triangle contact (Box3D `b3TriangleFeature`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum TriangleFeature {
    #[default]
    None,
    /// The triangle's face.
    TriangleFace,
    /// A face of the hull against the triangle.
    HullFace,
    /// The edge from corner 1 to corner 2.
    Edge1,
    /// The edge from corner 2 to corner 3.
    Edge2,
    /// The edge from corner 3 to corner 1.
    Edge3,
    Vertex1,
    Vertex2,
    Vertex3,
}

const EDGE_FEATURES: [TriangleFeature; 3] = [
    TriangleFeature::Edge1,
    TriangleFeature::Edge2,
    TriangleFeature::Edge3,
];

/// A triangle contact: its normal from the triangle to the shape, its point count, the feature
/// and, for spheres, the squared distance from the sphere's centre.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TriangleContact {
    pub normal: Vec3,
    pub count: usize,
    pub feature: TriangleFeature,
    pub squared_distance: f32,
}

/// Box3D `b3GetTriangleFeature`: the triangle corners in a GJK simplex name the feature.
fn triangle_feature(cache: &SimplexCache) -> TriangleFeature {
    let mask = cache.index_a[..cache.count as usize]
        .iter()
        .fold(0usize, |mask, &i| mask | (1 << i));
    [
        TriangleFeature::None,
        TriangleFeature::Vertex1,
        TriangleFeature::Vertex2,
        TriangleFeature::Edge1,
        TriangleFeature::Vertex3,
        TriangleFeature::Edge3,
        TriangleFeature::Edge2,
        TriangleFeature::TriangleFace,
    ][mask]
}

/// Box3D `b3ClosestPointOnTriangle`: the point of triangle `a b c` closest to `q`, and the
/// feature it lies on.
pub(crate) fn closest_point_on_triangle(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    q: Vec3,
) -> (Vec3, TriangleFeature) {
    let ab = b - a;
    let ac = c - a;
    let aq = q - a;
    let d1 = ab.dot(aq);
    let d2 = ac.dot(aq);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, TriangleFeature::Vertex1);
    }
    let bq = q - b;
    let d3 = ab.dot(bq);
    let d4 = ac.dot(bq);
    if d3 > 0.0 && d4 <= d3 {
        return (b, TriangleFeature::Vertex2);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return (a + ab * t, TriangleFeature::Edge1);
    }
    let cq = q - c;
    let d5 = ab.dot(cq);
    let d6 = ac.dot(cq);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, TriangleFeature::Vertex3);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return (a + ac * t, TriangleFeature::Edge3);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 >= d3 && d5 >= d6 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * t, TriangleFeature::Edge2);
    }
    let t1 = vb / (va + vb + vc);
    let t2 = vc / (va + vb + vc);
    (a + ab * t1 + ac * t2, TriangleFeature::TriangleFace)
}

/// Box3D `b3CollideTriangleAndSphere`: one point midway between the sphere and its closest
/// point on the triangle, if the centre is in front.
pub(crate) fn collide_triangle_and_sphere(
    points: &mut [LocalPoint],
    triangle: &[Vec3; 3],
    sphere: &Sphere,
) -> TriangleContact {
    let mut contact = TriangleContact::default();
    if points.is_empty() {
        return contact;
    }
    let center = sphere.center;
    let [v1, v2, v3] = *triangle;
    let plane = Plane::from_points(v1, v2, v3);
    if plane.separation(center) < 0.0 {
        // Behind the triangle.
        return contact;
    }
    let radius = sphere.radius;
    let (closest, feature) = closest_point_on_triangle(v1, v2, v3, center);
    let squared_distance = (closest - center).length_squared();
    let max_distance = radius + SPECULATIVE_DISTANCE;
    if squared_distance > max_distance * max_distance {
        return contact;
    }
    let distance = squared_distance.sqrt();
    let normal = if distance * distance > 1000.0 * f32::MIN_POSITIVE {
        (center - closest) * (1.0 / distance)
    } else {
        normalize((v2 - v1).cross(v3 - v1))
    };
    points[0] = LocalPoint {
        point: ((center - normal * radius) + closest) * 0.5,
        separation: distance - radius,
        pair: FeaturePair::SINGLE,
    };
    contact.normal = normal;
    contact.count = 1;
    contact.feature = feature;
    contact.squared_distance = squared_distance;
    contact
}

/// Box3D `b3ClipSegmentToTriangleFace`: the segment clipped to the triangle's side planes;
/// false when it leaves them.
fn clip_segment_to_triangle_face(
    segment: &mut [ClipVertex; 2],
    points: &[Vec3; 3],
    plane: &Plane,
) -> bool {
    let mut vertex1 = points[2];
    for &vertex2 in points {
        let tangent = normalize(vertex2 - vertex1);
        let binormal = tangent.cross(plane.normal);
        let clip_plane = Plane::from_normal_and_point(binormal, vertex1);
        let (p1, p2) = (segment[0], segment[1]);
        let distance1 = clip_plane.separation(p1.position);
        let distance2 = clip_plane.separation(p2.position);
        let mut count = 0;
        if distance1 <= 0.0 {
            segment[count] = p1;
            count += 1;
        }
        if distance2 <= 0.0 {
            segment[count] = p2;
            count += 1;
        }
        if (distance1 > 0.0) != (distance2 > 0.0) {
            let t = distance1 / (distance1 - distance2);
            segment[count].position = lerp(p1.position, p2.position, t);
            segment[count].pair = if distance1 > 0.0 { p1.pair } else { p2.pair };
            count += 1;
        }
        if count != 2 {
            return false;
        }
        vertex1 = vertex2;
    }
    true
}

fn capsule_segment(capsule: &Capsule) -> [ClipVertex; 2] {
    [
        ClipVertex {
            position: capsule.center1,
            separation: 0.0,
            pair: FeaturePair::new(SHAPE_A, 0, SHAPE_A, 0),
        },
        ClipVertex {
            position: capsule.center2,
            separation: 0.0,
            pair: FeaturePair::new(SHAPE_A, 1, SHAPE_A, 1),
        },
    ]
}

/// Two face points of a clipped capsule segment, halfway between the capsule and the plane.
fn segment_face_points(
    points: &mut [LocalPoint],
    segment: &[ClipVertex; 2],
    plane: &Plane,
    radius: f32,
) -> TriangleContact {
    for (point, vertex) in points.iter_mut().zip(segment) {
        let distance = plane.separation(vertex.position);
        *point = LocalPoint {
            point: vertex.position - plane.normal * (0.5 * (radius + distance)),
            separation: distance - radius,
            pair: vertex.pair,
        };
    }
    TriangleContact {
        normal: plane.normal,
        count: 2,
        feature: TriangleFeature::TriangleFace,
        squared_distance: 0.0,
    }
}

/// A separating axis candidate of the triangle builders.
#[derive(Debug, Clone, Copy)]
struct Axis {
    normal: Vec3,
    separation: f32,
    index_a: i32,
    index_b: i32,
    kind: u8,
}

/// Box3D `b3QueryTriangleFaceAndCapsule`.
fn query_triangle_face_and_capsule(plane: &Plane, capsule: &Capsule) -> Axis {
    let separation1 = plane.separation(capsule.center1);
    let separation2 = plane.separation(capsule.center2);
    let (separation, index_b) = if separation1 < separation2 {
        (separation1, 0)
    } else {
        (separation2, 1)
    };
    Axis {
        normal: plane.normal,
        separation,
        index_a: 0,
        index_b,
        kind: AXIS_FACE_A,
    }
}

/// Box3D `b3QueryTriangleAndCapsuleEdges`: the edge axis of least penetration, each triangle
/// edge seen as a zero-area face with a side normal so the axis points out of the triangle.
fn query_triangle_and_capsule_edges(
    vertices: &[Vec3; 3],
    plane: &Plane,
    capsule: &Capsule,
) -> Axis {
    let (p1, p2) = (capsule.center1, capsule.center2);
    let capsule_edge = p2 - p1;
    let mut best = Axis {
        normal: Vec3::ZERO,
        separation: -f32::MAX,
        index_a: -1,
        index_b: -1,
        kind: AXIS_EDGE_PAIR,
    };
    let mut edge_index = 2;
    let a = capsule_edge.dot(plane.normal);
    let mut v1 = vertices[2];
    for (index, &v2) in vertices.iter().enumerate() {
        let triangle_edge = v2 - v1;
        let side_normal = normalize(triangle_edge.cross(plane.normal));
        let b = capsule_edge.dot(side_normal);
        if a * a + b * b < SQUARED_TOLERANCE * capsule_edge.length_squared() {
            // Parallel to the edge: the face contact handles it.
            v1 = v2;
            edge_index = index;
            continue;
        }
        let axis = if a * b <= 0.0 {
            lerp(side_normal, plane.normal, b / (b - a))
        } else {
            lerp(side_normal, -plane.normal, b / (a + b))
        };
        let axis = normalize(axis);
        let separation = axis.dot(p1 - v1);
        if separation > best.separation {
            best = Axis {
                normal: axis,
                separation,
                index_a: edge_index as i32,
                index_b: 0,
                kind: AXIS_EDGE_PAIR,
            };
        }
        v1 = v2;
        edge_index = index;
    }
    best
}

/// Box3D `b3BuildTriangleAndCapsuleEdgeContact`.
fn triangle_and_capsule_edge_contact(
    points: &mut [LocalPoint],
    triangle: &[Vec3; 3],
    plane: &Plane,
    capsule: &Capsule,
    query: &Axis,
) -> TriangleContact {
    let mut contact = TriangleContact::default();
    let (p1, p2) = (capsule.center1, capsule.center2);
    let capsule_edge = p2 - p1;
    let index_a = query.index_a as usize;
    let v1 = triangle[index_a];
    let v2 = triangle[(index_a + 1) % 3];
    let triangle_edge = v2 - v1;
    let side_normal = normalize(triangle_edge.cross(plane.normal));
    let a = capsule_edge.dot(plane.normal);
    let b = capsule_edge.dot(side_normal);
    if a * a + b * b < SQUARED_TOLERANCE * capsule_edge.length_squared() {
        return contact;
    }
    let normal = query.normal;
    let result = line_distance(v1, triangle_edge, p1, capsule_edge);
    if !result.is_within_segments() {
        // The closest points lie beyond the ends.
        return contact;
    }
    points[0] = LocalPoint {
        point: lerp(result.point1, result.point2 - normal * capsule.radius, 0.5),
        separation: normal.dot(p1 - v1) - capsule.radius,
        pair: FeaturePair::new(SHAPE_A, index_a, SHAPE_B, query.index_b as usize),
    };
    contact.normal = normal;
    contact.count = 1;
    contact.feature = EDGE_FEATURES[index_a];
    contact
}

/// Box3D `b3CollideTriangleAndCapsule`: two points where the capsule lies along the face,
/// else one from the closest points, else (deep) the face or edge axis of least penetration.
pub(crate) fn collide_triangle_and_capsule(
    points: &mut [LocalPoint],
    triangle: &[Vec3; 3],
    capsule: &Capsule,
    cache: &mut SimplexCache,
) -> TriangleContact {
    let mut contact = TriangleContact::default();
    if points.len() < 2 {
        return contact;
    }
    let [v1, v2, v3] = *triangle;
    let plane = Plane::from_points(v1, v2, v3);
    let capsule_center = lerp(capsule.center1, capsule.center2, 0.5);
    if plane.separation(capsule_center) < 0.0 {
        // Behind the triangle.
        return contact;
    }
    let capsule_points = [capsule.center1, capsule.center2];
    let output = shape_distance(
        &ShapeProxy {
            points: triangle,
            radius: 0.0,
        },
        &ShapeProxy {
            points: &capsule_points,
            radius: 0.0,
        },
        &Transform::IDENTITY,
        false,
        cache,
    );
    let radius = capsule.radius;
    if output.distance > radius + SPECULATIVE_DISTANCE {
        // Apart; the cache persists.
        return contact;
    }

    if output.distance > 100.0 * f32::EPSILON {
        // Shallow.
        let delta = normalize(output.point_b - output.point_a);
        // Two points where the closest points line up with the face normal.
        if plane.normal.dot(delta).abs() > 0.2 {
            let mut segment = capsule_segment(capsule);
            if clip_segment_to_triangle_face(&mut segment, triangle, &plane) {
                return segment_face_points(points, &segment, &plane, radius);
            }
        }
        points[0] = LocalPoint {
            point: lerp(output.point_a, output.point_b - delta * radius, 0.5),
            separation: output.distance - radius,
            pair: FeaturePair::SINGLE,
        };
        contact.normal = delta;
        contact.count = 1;
        contact.feature = triangle_feature(cache);
        return contact;
    }

    // Deep.
    let face_query = query_triangle_face_and_capsule(&plane, capsule);
    if face_query.separation > radius {
        return contact;
    }
    let edge_query = query_triangle_and_capsule_edges(triangle, &plane, capsule);
    if edge_query.separation > radius {
        return contact;
    }
    let mut face_separation = face_query.separation - radius;
    let mut segment = capsule_segment(capsule);
    if clip_segment_to_triangle_face(&mut segment, triangle, &plane) {
        let distance1 = plane.separation(segment[0].position);
        let distance2 = plane.separation(segment[1].position);
        if distance1 <= SPECULATIVE_DISTANCE + radius || distance2 <= SPECULATIVE_DISTANCE + radius
        {
            contact = segment_face_points(points, &segment, &plane, radius);
            // The clipped separation.
            face_separation = points[0].separation.min(points[1].separation);
        }
    }
    if edge_query.index_a < 0 {
        return contact;
    }
    // The face contact may miss the axis of least penetration: the edge contact replaces it
    // then, or when it is clearly better.
    let edge_separation = edge_query.separation - radius;
    if contact.count == 0 || edge_separation > face_separation + LINEAR_SLOP {
        // An edge contact that makes no point leaves the face contact (Box3D writes into the
        // same manifold).
        let edge =
            triangle_and_capsule_edge_contact(points, triangle, &plane, capsule, &edge_query);
        if edge.count > 0 {
            contact = edge;
        }
    }
    contact
}

/// The triangle as the hull builder sees it (Box3D `b3TriangleData`).
struct TriangleData {
    points: [Vec3; 3],
    edges: [Vec3; 3],
    plane: Plane,
}

/// Box3D `b3QueryTriangleFace`: the triangle's plane against the hull's deepest vertex.
fn query_triangle_face(triangle: &TriangleData, hull: &HullView<'_>) -> Axis {
    let plane = triangle.plane;
    let vertex = hull.support_vertex(-plane.normal);
    Axis {
        normal: plane.normal,
        separation: plane.separation(hull.points[vertex]),
        index_a: 0,
        index_b: vertex as i32,
        kind: AXIS_FACE_A,
    }
}

/// Box3D `b3GetTriangleSupport`.
fn triangle_support(points: &[Vec3; 3], direction: Vec3) -> usize {
    let mut index = 0;
    let distance = points[0].dot(direction);
    let d = points[1].dot(direction);
    let distance = if d > distance {
        index = 1;
        d
    } else {
        distance
    };
    if points[2].dot(direction) > distance {
        return 2;
    }
    index
}

/// Box3D `b3QueryHullFace`: each hull face against the triangle's deepest corner; the normal
/// points from the triangle to the hull.
fn query_hull_face(triangle: &TriangleData, hull: &HullView<'_>) -> Axis {
    let mut best = Axis {
        normal: Vec3::ZERO,
        separation: f32::NEG_INFINITY,
        index_a: -1,
        index_b: -1,
        kind: AXIS_FACE_B,
    };
    for (face, plane) in hull.planes.iter().enumerate() {
        let vertex = triangle_support(&triangle.points, -plane.normal);
        let separation = plane.separation(triangle.points[vertex]);
        if separation > best.separation {
            best = Axis {
                normal: plane.normal,
                separation,
                index_a: vertex as i32,
                index_b: face as i32,
                kind: AXIS_FACE_B,
            };
        }
    }
    best.normal = -best.normal;
    best
}

/// The axis normal to triangle edge `triangle_edge` and the hull edge between faces of normals
/// `normal1` and `normal2`, if the edges build a face of the Minkowski difference.
fn edge_axis(
    normal1: Vec3,
    normal2: Vec3,
    triangle_edge: Vec3,
    triangle_normal: Vec3,
    hull_edge: Vec3,
) -> Option<Vec3> {
    let cab = normal1.dot(triangle_edge);
    let dab = normal2.dot(triangle_edge);
    let bcd = triangle_normal.dot(hull_edge);
    if cab * dab >= 0.0 || cab * bcd <= 0.0 {
        return None;
    }
    // Nearly parallel edges give separations at the noise floor.
    if (cab * cab).max(dab * dab) < SQUARED_TOLERANCE * triangle_edge.length_squared() {
        return None;
    }
    // dot(normal1 + t * (normal2 - normal1), triangle_edge) = 0, out of the hull.
    Some(normalize(lerp(normal1, normal2, cab / (cab - dab))))
}

/// Box3D `b3QueryTriangleAndHullEdges`: the edge pair axis of least penetration; the normal
/// points from the triangle to the hull.
fn query_triangle_and_hull_edges(triangle: &TriangleData, hull: &HullView<'_>) -> Axis {
    let mut best = Axis {
        normal: Vec3::ZERO,
        separation: f32::NEG_INFINITY,
        index_a: -1,
        index_b: -1,
        kind: AXIS_EDGE_PAIR,
    };
    for i in (0..hull.edges.len()).step_by(2) {
        let (edge, twin) = (hull.edges[i], hull.edges[i + 1]);
        let hull_point = hull.points[edge.origin as usize];
        let hull_edge = hull.points[twin.origin as usize] - hull_point;
        let normal1 = hull.planes[edge.face as usize].normal;
        let normal2 = hull.planes[twin.face as usize].normal;
        for j in 0..3 {
            let Some(axis) = edge_axis(
                normal1,
                normal2,
                triangle.edges[j],
                triangle.plane.normal,
                hull_edge,
            ) else {
                continue;
            };
            let separation = axis.dot(triangle.points[j] - hull_point);
            if separation > best.separation {
                best = Axis {
                    normal: -axis,
                    separation,
                    index_a: j as i32,
                    index_b: i as i32,
                    kind: AXIS_EDGE_PAIR,
                };
            }
        }
    }
    best
}

/// Writes the clipped polygon's points within the speculative distance, at most
/// `points.len()`; returns their count and the smallest separation of all.
fn clipped_points(
    points: &mut [LocalPoint],
    polygon: &[ClipVertex],
    mut point: impl FnMut(&ClipVertex) -> LocalPoint,
) -> (usize, f32) {
    let count = polygon.len().min(points.len());
    let mut min_separation = f32::MAX;
    for (out, vertex) in points.iter_mut().zip(&polygon[..count]) {
        min_separation = min_separation.min(vertex.separation);
        *out = point(vertex);
    }
    (count, min_separation)
}

/// Box3D `b3CollideHullFace`: the triangle clipped to reference face `query.index_b` of the
/// hull. Returns the contact and the clipped separation.
fn collide_hull_face(
    points: &mut [LocalPoint],
    triangle: &TriangleData,
    hull: &HullView<'_>,
    query: &Axis,
    cache: &mut SatCache,
) -> (TriangleContact, f32) {
    let ref_face = query.index_b as usize;
    let ref_plane = hull.planes[ref_face];
    let [v1, v2, v3] = triangle.points;
    let mut input = [ClipVertex::default(); MAX_CLIP_POINTS];
    let mut output = [ClipVertex::default(); MAX_CLIP_POINTS];
    for (i, (v, pair)) in [
        (v1, FeaturePair::new(SHAPE_B, 2, SHAPE_B, 0)),
        (v2, FeaturePair::new(SHAPE_B, 0, SHAPE_B, 1)),
        (v3, FeaturePair::new(SHAPE_B, 1, SHAPE_B, 2)),
    ]
    .into_iter()
    .enumerate()
    {
        input[i] = ClipVertex {
            position: v,
            separation: ref_plane.separation(v),
            pair,
        };
    }
    let mut count = 3;
    let first = hull.face_edges[ref_face] as usize;
    let mut edge_index = first;
    loop {
        let edge = hull.edges[edge_index];
        let next_index = edge.next as usize;
        let vertex1 = hull.points[edge.origin as usize];
        let vertex2 = hull.points[hull.edges[next_index].origin as usize];
        let tangent = normalize(vertex2 - vertex1);
        let binormal = tangent.cross(ref_plane.normal);
        let clip_plane = Plane::from_normal_and_point(binormal, vertex1);
        count = clip_polygon(
            &mut output,
            &input[..count],
            &clip_plane,
            edge_index,
            &ref_plane,
        );
        if count < 3 {
            // A stale cache.
            *cache = SatCache::default();
            return (TriangleContact::default(), query.separation);
        }
        std::mem::swap(&mut input, &mut output);
        edge_index = next_index;
        if edge_index == first {
            break;
        }
    }
    let (count, min_separation) = clipped_points(points, &input[..count], |v| LocalPoint {
        // On the hull face, for better culling.
        point: v.position - ref_plane.normal * v.separation,
        separation: v.separation,
        pair: v.pair.flip(),
    });
    if min_separation > SPECULATIVE_DISTANCE {
        // A stale cache.
        *cache = SatCache::default();
        return (TriangleContact::default(), min_separation);
    }
    *cache = SatCache {
        separation: min_separation,
        kind: AXIS_FACE_B,
        index_a: query.index_a as u8,
        index_b: query.index_b as u8,
        hit: cache.hit,
    };
    (
        TriangleContact {
            normal: -ref_plane.normal,
            count,
            feature: TriangleFeature::HullFace,
            squared_distance: 0.0,
        },
        min_separation,
    )
}

/// Box3D `b3CollideTriangleFace`: the hull's incident face clipped to the triangle.
fn collide_triangle_face(
    points: &mut [LocalPoint],
    triangle: &TriangleData,
    hull: &HullView<'_>,
    query: &Axis,
    cache: &mut SatCache,
) -> (TriangleContact, f32) {
    let ref_plane = triangle.plane;
    let incident = find_incident_face(hull, ref_plane.normal, query.index_b as usize);
    let mut input = [ClipVertex::default(); MAX_CLIP_POINTS];
    let mut output = [ClipVertex::default(); MAX_CLIP_POINTS];
    let mut count = 0;
    let first = hull.face_edges[incident] as usize;
    let mut edge_index = first;
    loop {
        let next_index = hull.edges[edge_index].next as usize;
        let point = hull.points[hull.edges[next_index].origin as usize];
        input[count] = ClipVertex {
            position: point,
            separation: ref_plane.separation(point),
            pair: FeaturePair::new(SHAPE_B, edge_index, SHAPE_B, next_index),
        };
        count += 1;
        edge_index = next_index;
        if edge_index == first || count == MAX_CLIP_POINTS {
            break;
        }
    }
    for i in 0..3 {
        if count == 0 {
            break;
        }
        let side_normal = normalize(triangle.edges[i].cross(ref_plane.normal));
        let clip_plane = Plane::from_normal_and_point(side_normal, triangle.points[i]);
        count = clip_polygon(&mut output, &input[..count], &clip_plane, i, &ref_plane);
        std::mem::swap(&mut input, &mut output);
    }
    if count == 0 {
        // The face clipped away.
        *cache = SatCache::default();
        return (TriangleContact::default(), f32::MAX);
    }
    let (count, min_separation) = clipped_points(points, &input[..count], |v| LocalPoint {
        point: v.position,
        separation: v.separation,
        pair: v.pair,
    });
    if min_separation >= SPECULATIVE_DISTANCE {
        // Moved apart while reusing a cached axis.
        *cache = SatCache::default();
        return (TriangleContact::default(), min_separation);
    }
    *cache = SatCache {
        separation: min_separation,
        kind: AXIS_FACE_A,
        index_a: query.index_a as u8,
        index_b: query.index_b as u8,
        hit: cache.hit,
    };
    (
        TriangleContact {
            normal: ref_plane.normal,
            count,
            feature: TriangleFeature::TriangleFace,
            squared_distance: 0.0,
        },
        min_separation,
    )
}

/// Box3D `b3CollideTriangleAndHullEdges`: one point between the closest points of the edges.
fn collide_triangle_and_hull_edges(
    points: &mut [LocalPoint],
    triangle_point: Vec3,
    triangle_edge: Vec3,
    hull: &HullView<'_>,
    query: &Axis,
    cache: &mut SatCache,
) -> TriangleContact {
    let edge = hull.edges[query.index_b as usize];
    let twin = hull.edges[edge.twin as usize];
    let p_b = hull.points[edge.origin as usize];
    let q_b = hull.points[twin.origin as usize];
    let result = line_distance(triangle_point, triangle_edge, p_b, q_b - p_b);
    if points.is_empty() || !result.is_within_segments() {
        *cache = SatCache::default();
        return TriangleContact::default();
    }
    let separation = query.normal.dot(p_b - triangle_point);
    points[0] = LocalPoint {
        point: (result.point1 + result.point2) * 0.5,
        separation,
        pair: FeaturePair::new(
            SHAPE_A,
            query.index_a as usize,
            SHAPE_B,
            query.index_b as usize,
        ),
    };
    *cache = SatCache {
        separation,
        kind: AXIS_EDGE_PAIR,
        index_a: query.index_a as u8,
        index_b: query.index_b as u8,
        hit: cache.hit,
    };
    TriangleContact {
        normal: query.normal,
        count: 1,
        feature: EDGE_FEATURES[query.index_a as usize],
        squared_distance: 0.0,
    }
}

/// Box3D `b3CollideTriangleAndHull`, the triangle `v` in the hull's frame: the cached axis is
/// tried first; otherwise the triangle face, the hull faces and the edge pairs are queried,
/// the better face clipped (a hull face pushing down into the triangle is refused, to avoid
/// ghost contacts), an edge contact taken when clearly deeper, and GJK used when the axes give
/// no point.
pub(crate) fn collide_triangle_and_hull(
    points: &mut [LocalPoint],
    v: &[Vec3; 3],
    hull: &HullView<'_>,
    cache: &mut SatCache,
) -> TriangleContact {
    let none = TriangleContact::default();
    if points.len() < 4 {
        return none;
    }
    let [v1, v2, v3] = *v;
    let triangle_plane = Plane::from_points(v1, v2, v3);
    let offset = triangle_plane.separation(hull.center);
    if cache.kind == AXIS_BACKSIDE {
        // Hysteresis against jitter on wavy meshes.
        if (cache.separation - offset).abs() < LINEAR_SLOP {
            return none;
        }
        cache.kind = AXIS_INVALID;
    }
    if offset < -LINEAR_SLOP {
        // Behind the triangle; remember the offset.
        cache.kind = AXIS_BACKSIDE;
        cache.separation = offset;
        return none;
    }
    let triangle = TriangleData {
        points: *v,
        edges: [v2 - v1, v3 - v2, v1 - v3],
        plane: triangle_plane,
    };
    let speculative_distance = SPECULATIVE_DISTANCE;
    cache.hit = 1;

    match cache.kind {
        AXIS_FACE_A => {
            let vertex = hull.support_vertex(-triangle_plane.normal);
            let separation = triangle_plane.separation(hull.points[vertex]);
            if separation > speculative_distance {
                // Still apart.
                return none;
            }
            let query = Axis {
                normal: triangle_plane.normal,
                separation,
                index_a: i32::from(cache.index_a),
                index_b: vertex as i32,
                kind: AXIS_FACE_A,
            };
            let mut local = *cache;
            let (contact, clipped) =
                collide_triangle_face(points, &triangle, hull, &query, &mut local);
            if contact.count > 0 && (cache.separation - clipped).abs() < LINEAR_SLOP {
                return contact;
            }
            *cache = SatCache::default();
        }
        AXIS_FACE_B => {
            let plane = hull.planes[cache.index_b as usize];
            let mut vertex = 0;
            let mut distance = -v1.dot(plane.normal);
            for (i, p) in triangle.points.iter().enumerate().skip(1) {
                let d = -p.dot(plane.normal);
                if d > distance {
                    distance = d;
                    vertex = i;
                }
            }
            let separation = plane.separation(triangle.points[vertex]);
            if separation > speculative_distance {
                return none;
            }
            // A deep overlap may invalidate the cache.
            if separation >= -2.0 * LINEAR_SLOP {
                let query = Axis {
                    normal: -plane.normal,
                    separation,
                    index_a: vertex as i32,
                    index_b: i32::from(cache.index_b),
                    kind: AXIS_FACE_B,
                };
                let mut local = *cache;
                let (contact, clipped) =
                    collide_hull_face(points, &triangle, hull, &query, &mut local);
                if contact.count > 0 && (cache.separation - clipped).abs() < LINEAR_SLOP {
                    return contact;
                }
            }
            *cache = SatCache::default();
        }
        AXIS_EDGE_PAIR => {
            let index_a = cache.index_a as usize;
            let index_b = cache.index_b as usize;
            let (edge, twin) = (hull.edges[index_b], hull.edges[index_b + 1]);
            let hull_point = hull.points[edge.origin as usize];
            let hull_edge = hull.points[twin.origin as usize] - hull_point;
            let axis = edge_axis(
                hull.planes[edge.face as usize].normal,
                hull.planes[twin.face as usize].normal,
                triangle.edges[index_a],
                triangle_plane.normal,
                hull_edge,
            );
            if let Some(axis) = axis {
                let separation = axis.dot(triangle.points[index_a] - hull_point);
                if separation > speculative_distance {
                    return none;
                }
                if (cache.separation - separation).abs() < LINEAR_SLOP {
                    let query = Axis {
                        normal: -axis,
                        separation,
                        index_a: index_a as i32,
                        index_b: index_b as i32,
                        kind: AXIS_EDGE_PAIR,
                    };
                    let mut local = *cache;
                    let contact = collide_triangle_and_hull_edges(
                        points,
                        triangle.points[index_a],
                        triangle.edges[index_a],
                        hull,
                        &query,
                        &mut local,
                    );
                    if contact.count > 0 {
                        return contact;
                    }
                }
            }
            *cache = SatCache::default();
        }
        _ => {}
    }

    // A cache miss.
    cache.hit = 0;
    let face_a = query_triangle_face(&triangle, hull);
    if face_a.separation > speculative_distance {
        *cache = sat_cache(&face_a);
        return none;
    }
    let face_b = query_hull_face(&triangle, hull);
    if face_b.separation > speculative_distance {
        *cache = sat_cache(&face_b);
        return none;
    }
    let edge = query_triangle_and_hull_edges(&triangle, hull);
    if edge.separation > speculative_distance {
        *cache = sat_cache(&edge);
        return none;
    }
    // A hull face opposing the triangle face would make ghost contacts.
    let pushing_down = face_b.normal.dot(triangle_plane.normal) < -0.25;
    let (mut contact, clip_separation) = if face_b.separation >= face_a.separation && !pushing_down
    {
        collide_hull_face(points, &triangle, hull, &face_b, cache)
    } else {
        collide_triangle_face(points, &triangle, hull, &face_a, cache)
    };
    if edge.index_a >= 0 {
        // With aligned axes the edge separation can be garbage, and a face axis with positive
        // separation may give no points.
        let max_face_separation = face_a.separation.max(face_b.separation);
        if (contact.count == 0 && edge.separation > max_face_separation)
            || (contact.count == 1 && edge.separation > clip_separation + LINEAR_SLOP)
        {
            let j = edge.index_a as usize;
            contact = collide_triangle_and_hull_edges(
                points,
                triangle.points[j],
                triangle.edges[j],
                hull,
                &edge,
                cache,
            );
        }
    }
    // The speculative distance can leave the axes without points: GJK keeps the shapes from
    // tunnelling.
    if contact.count == 0 {
        let mut simplex = SimplexCache::default();
        let output = shape_distance(
            &ShapeProxy {
                points: v,
                radius: 0.0,
            },
            &ShapeProxy {
                points: hull.points,
                radius: 0.0,
            },
            &Transform::IDENTITY,
            false,
            &mut simplex,
        );
        if output.distance > 0.0 {
            points[0] = LocalPoint {
                point: output.point_b,
                separation: output.distance,
                pair: FeaturePair::SINGLE,
            };
            contact = TriangleContact {
                normal: output.normal,
                count: 1,
                feature: triangle_feature(&simplex),
                squared_distance: 0.0,
            };
        }
        *cache = SatCache::default();
    }
    contact
}

/// The cache of a separating axis.
fn sat_cache(axis: &Axis) -> SatCache {
    SatCache {
        separation: axis.separation,
        kind: axis.kind,
        index_a: axis.index_a as u8,
        index_b: axis.index_b as u8,
        hit: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UP: [Vec3; 3] = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(-1.0, 0.0, 2.0),
        Vec3::new(2.0, 0.0, -1.0),
    ];

    #[test]
    fn spheres_collide_from_the_front_only() {
        let mut points = [LocalPoint::default(); 4];
        let above = Sphere {
            center: Vec3::new(0.0, 0.21, 0.0),
            radius: 0.2,
        };
        let contact = collide_triangle_and_sphere(&mut points, &UP, &above);
        assert_eq!(contact.count, 1);
        assert_eq!(contact.feature, TriangleFeature::TriangleFace);
        assert!((points[0].separation - 0.01).abs() < 1e-6);
        let below = Sphere {
            center: Vec3::new(0.0, -0.1, 0.0),
            radius: 0.2,
        };
        assert_eq!(
            collide_triangle_and_sphere(&mut points, &UP, &below).count,
            0
        );
    }

    #[test]
    fn closest_features_name_the_region() {
        let [a, b, c] = UP;
        let cases = [
            (Vec3::new(-2.0, 1.0, -2.0), TriangleFeature::Vertex1),
            (Vec3::new(-2.0, 1.0, 0.0), TriangleFeature::Edge1),
            (Vec3::new(0.0, 1.0, 0.0), TriangleFeature::TriangleFace),
            (Vec3::new(0.0, 1.0, -2.0), TriangleFeature::Edge3),
            (Vec3::new(2.0, 1.0, 2.0), TriangleFeature::Edge2),
        ];
        for (q, feature) in cases {
            assert_eq!(closest_point_on_triangle(a, b, c, q).1, feature, "{q:?}");
        }
    }
}
