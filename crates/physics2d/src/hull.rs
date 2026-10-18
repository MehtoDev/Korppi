//! Convex hulls (Box2D v3.2 `hull.c`): the point sets polygons are made from.
//!
//! [`Hull2d::compute`] is Box2D's quickhull: it welds points closer than half a linear slop,
//! keeps the first of each, starts from the point farthest from the centre of the points'
//! bounds, and drops points within two linear slops of a hull edge, so the hull has no
//! collinear points. It fails for fewer than 3 or more than [`MAX_POLYGON_VERTICES`] points
//! and for points that are all close together or on one line. [`Hull2d::new`] takes points
//! that already form a hull, checked by [`Hull2d::validate`].

use crate::distance::LINEAR_SLOP;
use crate::math::{cross, distance_squared, normalize};
use crate::shape::MAX_POLYGON_VERTICES;
use korppi_math::Vec2;

/// A convex hull of 3 to [`MAX_POLYGON_VERTICES`] points, counter-clockwise, without collinear
/// points (Box2D `b2Hull`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hull2d {
    points: [Vec2; MAX_POLYGON_VERTICES],
    count: usize,
}

/// Up to [`MAX_POLYGON_VERTICES`] points on the stack.
#[derive(Clone, Copy)]
struct Points {
    points: [Vec2; MAX_POLYGON_VERTICES],
    count: usize,
}

impl Points {
    const EMPTY: Self = Self {
        points: [Vec2::ZERO; MAX_POLYGON_VERTICES],
        count: 0,
    };

    fn push(&mut self, point: Vec2) {
        self.points[self.count] = point;
        self.count += 1;
    }

    fn as_slice(&self) -> &[Vec2] {
        &self.points[..self.count]
    }
}

impl Hull2d {
    /// The convex hull of `points` (Box2D `b2ComputeHull`), or `None` if it fails.
    #[must_use]
    pub fn compute(points: &[Vec2]) -> Option<Self> {
        Self::compute_with_slop(points, LINEAR_SLOP)
    }

    /// [`Self::compute`] for a world whose linear slop is `linear_slop`
    /// ([`WorldConfig2d::linear_slop`](crate::config::WorldConfig2d::linear_slop)).
    #[must_use]
    pub fn compute_with_slop(points: &[Vec2], linear_slop: f32) -> Option<Self> {
        if points.len() < 3 || points.len() > MAX_POLYGON_VERTICES {
            return None;
        }

        // Weld points closer than half a slop, keeping the first, and find the bounds.
        let tolerance_squared = 0.25 * linear_slop * linear_slop;
        let mut ps = Points::EMPTY;
        let (mut lower, mut upper) = (Vec2::splat(f32::MAX), Vec2::splat(-f32::MAX));
        for &point in points {
            lower = lower.min(point);
            upper = upper.max(point);
            if ps
                .as_slice()
                .iter()
                .all(|&p| distance_squared(point, p) >= tolerance_squared)
            {
                ps.push(point);
            }
        }
        if ps.count < 3 {
            return None;
        }

        // The hull starts from the point farthest from the centre of the bounds, then the one
        // farthest from it; both leave the working set.
        let center = Vec2::new(0.5 * (lower.x + upper.x), 0.5 * (lower.y + upper.y));
        let p1 = take_farthest(&mut ps, center);
        let p2 = take_farthest(&mut ps, p1);

        // Split the rest into the points right and left of p1-p2, skipping those near it.
        let mut right = Points::EMPTY;
        let mut left = Points::EMPTY;
        let e = normalize(p2 - p1);
        for &p in ps.as_slice() {
            let d = cross(p - p1, e);
            if d >= 2.0 * linear_slop {
                right.push(p);
            } else if d <= -2.0 * linear_slop {
                left.push(p);
            }
        }
        let hull1 = recurse_hull(p1, p2, right.as_slice(), linear_slop);
        let hull2 = recurse_hull(p2, p1, left.as_slice(), linear_slop);
        if hull1.count == 0 && hull2.count == 0 {
            // All points collinear.
            return None;
        }

        // Stitch the hulls together counter-clockwise.
        let mut hull = Points::EMPTY;
        hull.push(p1);
        hull1.as_slice().iter().for_each(|&p| hull.push(p));
        hull.push(p2);
        hull2.as_slice().iter().for_each(|&p| hull.push(p));

        // Merge collinear points.
        let mut searching = true;
        while searching && hull.count > 2 {
            searching = false;
            for i in 0..hull.count {
                let s1 = hull.points[i];
                let s2 = hull.points[(i + 1) % hull.count];
                let s3 = hull.points[(i + 2) % hull.count];
                let r = normalize(s3 - s1);
                if cross(s2 - s1, r) <= 2.0 * linear_slop {
                    // Remove the midpoint and search again.
                    let i2 = (i + 1) % hull.count;
                    hull.points.copy_within(i2 + 1..hull.count, i2);
                    hull.count -= 1;
                    searching = true;
                    break;
                }
            }
        }
        (hull.count >= 3).then_some(Self {
            points: hull.points,
            count: hull.count,
        })
    }

    /// `points` as a hull, in their order, if they form one (see [`Self::validate`]).
    #[must_use]
    pub fn new(points: &[Vec2]) -> Option<Self> {
        Self::validate(points).then(|| {
            let mut hull = Points::EMPTY;
            points.iter().for_each(|&p| hull.push(p));
            Self {
                points: hull.points,
                count: hull.count,
            }
        })
    }

    /// Whether `points` form a hull (Box2D `b2ValidateHull`): 3 to [`MAX_POLYGON_VERTICES`]
    /// points, every point behind every edge it is not on (so convex and counter-clockwise),
    /// and no point within a linear slop of the line through its neighbours.
    #[must_use]
    pub fn validate(points: &[Vec2]) -> bool {
        Self::validate_with_slop(points, LINEAR_SLOP)
    }

    /// [`Self::validate`] for a world whose linear slop is `linear_slop`.
    #[must_use]
    pub fn validate_with_slop(points: &[Vec2], linear_slop: f32) -> bool {
        let count = points.len();
        if !(3..=MAX_POLYGON_VERTICES).contains(&count) {
            return false;
        }
        for i1 in 0..count {
            let i2 = if i1 < count - 1 { i1 + 1 } else { 0 };
            let p = points[i1];
            let e = normalize(points[i2] - p);
            let outside = (0..count)
                .filter(|&j| j != i1 && j != i2)
                .any(|j| cross(points[j] - p, e) >= 0.0);
            if outside {
                return false;
            }
        }
        (0..count).all(|i| {
            let p1 = points[i];
            let p2 = points[(i + 1) % count];
            let p3 = points[(i + 2) % count];
            cross(p2 - p1, normalize(p3 - p1)) > linear_slop
        })
    }

    /// The hull's points, counter-clockwise.
    #[must_use]
    pub fn points(&self) -> &[Vec2] {
        &self.points[..self.count]
    }
}

/// Removes and returns the point of `ps` farthest from `from`, the first of equals; the last
/// point takes its place (Box2D's working set).
fn take_farthest(ps: &mut Points, from: Vec2) -> Vec2 {
    let mut best = 0;
    let mut best_distance = distance_squared(from, ps.points[0]);
    for i in 1..ps.count {
        let d = distance_squared(from, ps.points[i]);
        if d > best_distance {
            best = i;
            best_distance = d;
        }
    }
    let point = ps.points[best];
    ps.points[best] = ps.points[ps.count - 1];
    ps.count -= 1;
    point
}

/// The hull of `ps` right of the edge `p1 -> p2`, without `p1` and `p2` (Box2D
/// `b2RecurseHull`).
fn recurse_hull(p1: Vec2, p2: Vec2, ps: &[Vec2], linear_slop: f32) -> Points {
    let mut hull = Points::EMPTY;
    if ps.is_empty() {
        return hull;
    }
    let e = normalize(p2 - p1);

    // Keep the points right of e and find the farthest.
    let mut right = Points::EMPTY;
    let mut best = 0;
    let mut best_distance = cross(ps[0] - p1, e);
    if best_distance > 0.0 {
        right.push(ps[0]);
    }
    for (i, &p) in ps.iter().enumerate().skip(1) {
        let distance = cross(p - p1, e);
        if distance > best_distance {
            best = i;
            best_distance = distance;
        }
        if distance > 0.0 {
            right.push(p);
        }
    }
    if best_distance < 2.0 * linear_slop {
        return hull;
    }

    let best_point = ps[best];
    let hull1 = recurse_hull(p1, best_point, right.as_slice(), linear_slop);
    let hull2 = recurse_hull(best_point, p2, right.as_slice(), linear_slop);
    hull1.as_slice().iter().for_each(|&p| hull.push(p));
    hull.push(best_point);
    hull2.as_slice().iter().for_each(|&p| hull.push(p));
    hull
}
