//! Box3D's quickhull builder (`hull.c`, `b3HullBuilder`), on index arenas.
//!
//! The builder grows a hull from an initial tetrahedron by adding the farthest outside point,
//! deleting the faces it sees, closing the horizon with a cone of new faces and merging the
//! cone faces that are concave, coplanar or flipped. Box3D links its vertices, faces and
//! conflict lists with intrusive pointer lists; here they are indices into three arenas, with the
//! same list orders and the same free-list reuse, so that the finished hull numbers its
//! vertices, half-edges and faces as Box3D's does.

use crate::hull::{HalfEdge, Plane};
use korppi_math::Vec3;

const NULL: u32 = u32::MAX;
const MARK_VISIBLE: u8 = 0;
const MARK_DELETE: u8 = 1;

/// The final hull is indexed with `u8`, so no count may reach 256.
const MAX_COUNT: usize = 256;

#[derive(Debug, Clone, Copy)]
struct Vertex {
    prev: u32,
    next: u32,
    conflict_face: u32,
    position: Vec3,
    final_index: i32,
    reachable: bool,
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    prev: u32,
    next: u32,
    origin: u32,
    face: u32,
    twin: u32,
    final_index: i32,
}

#[derive(Debug, Clone, Copy)]
struct Face {
    prev: u32,
    next: u32,
    edge: u32,
    mark: u8,
    area: f32,
    plane: Plane,
    centroid: Vec3,
    max_conflict_distance: f32,
    /// The sentinel of the face's conflict list, a node of the vertex arena.
    conflict_head: u32,
    max_conflict: u32,
    final_index: i32,
    flipped: bool,
}

#[derive(Debug, Clone, Copy)]
struct HorizonFrame {
    start_edge: u32,
    edge: u32,
    started: bool,
}

/// The topology a build produces, in Box3D's order.
pub(crate) struct BuiltHull {
    pub points: Vec<Vec3>,
    pub planes: Vec<Plane>,
    pub edges: Vec<HalfEdge>,
    pub face_edges: Vec<u8>,
    pub vertex_edges: Vec<u8>,
}

struct Builder {
    tolerance: f32,
    min_radius: f32,
    min_outside: f32,
    interior_point: Vec3,
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
    orphaned_head: u32,
    vertex_head: u32,
    face_head: u32,
    edge_free: u32,
    face_free: u32,
    horizon: Vec<u32>,
    cone: Vec<u32>,
    merged_faces: Vec<u32>,
    horizon_stack: Vec<HorizonFrame>,
    final_vertex_count: usize,
    final_half_edge_count: usize,
    final_face_count: usize,
}

impl Builder {
    fn new() -> Self {
        let mut b = Self {
            tolerance: 0.0,
            min_radius: 0.0,
            min_outside: 0.0,
            interior_point: Vec3::ZERO,
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            orphaned_head: 0,
            vertex_head: 0,
            face_head: 0,
            edge_free: NULL,
            face_free: NULL,
            horizon: Vec::new(),
            cone: Vec::new(),
            merged_faces: Vec::new(),
            horizon_stack: Vec::new(),
            final_vertex_count: 0,
            final_half_edge_count: 0,
            final_face_count: 0,
        };
        b.orphaned_head = b.vertex_sentinel();
        b.vertex_head = b.vertex_sentinel();
        b.face_head = b.faces.len() as u32;
        b.faces.push(Face {
            prev: b.face_head,
            next: b.face_head,
            edge: NULL,
            mark: MARK_VISIBLE,
            area: 0.0,
            plane: Plane {
                normal: Vec3::ZERO,
                offset: 0.0,
            },
            centroid: Vec3::ZERO,
            max_conflict_distance: 0.0,
            conflict_head: NULL,
            max_conflict: NULL,
            final_index: -1,
            flipped: false,
        });
        b
    }

    // ---------------------------------------------------------------- vertex lists

    fn vertex_sentinel(&mut self) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(Vertex {
            prev: index,
            next: index,
            conflict_face: NULL,
            position: Vec3::ZERO,
            final_index: -1,
            reachable: false,
        });
        index
    }

    fn v_contains(&self, node: u32) -> bool {
        let v = &self.vertices[node as usize];
        v.prev != NULL && v.next != NULL
    }

    /// Inserts `node` before `place`.
    fn v_insert(&mut self, node: u32, place: u32) {
        debug_assert!(!self.v_contains(node) && self.v_contains(place));
        let prev = self.vertices[place as usize].prev;
        self.vertices[node as usize].prev = prev;
        self.vertices[node as usize].next = place;
        self.vertices[prev as usize].next = node;
        self.vertices[place as usize].prev = node;
    }

    fn v_remove(&mut self, node: u32) {
        let Vertex { prev, next, .. } = self.vertices[node as usize];
        self.vertices[prev as usize].next = next;
        self.vertices[next as usize].prev = prev;
        self.vertices[node as usize].prev = NULL;
        self.vertices[node as usize].next = NULL;
    }

    /// Box3D `b3QHList_PushBack`, which inserts before the list's last node (`Insert(node,
    /// head->prev)`), not after it: the first node pushed stays last. The order is Box3D's.
    fn v_push_back(&mut self, head: u32, node: u32) {
        let last = self.vertices[head as usize].prev;
        self.v_insert(node, last);
    }

    fn v_empty(&self, head: u32) -> bool {
        self.vertices[head as usize].next == head
    }

    // ---------------------------------------------------------------- face lists

    fn f_contains(&self, face: u32) -> bool {
        let f = &self.faces[face as usize];
        f.prev != NULL && f.next != NULL
    }

    /// Box3D `b3QHList_PushBack` on the face list: before the last face (see
    /// [`Self::v_push_back`]).
    fn f_push_back(&mut self, face: u32) {
        let place = self.faces[self.face_head as usize].prev;
        let prev = self.faces[place as usize].prev;
        self.faces[face as usize].prev = prev;
        self.faces[face as usize].next = place;
        self.faces[prev as usize].next = face;
        self.faces[place as usize].prev = face;
    }

    fn f_remove(&mut self, face: u32) {
        let (prev, next) = (
            self.faces[face as usize].prev,
            self.faces[face as usize].next,
        );
        self.faces[prev as usize].next = next;
        self.faces[next as usize].prev = prev;
        self.faces[face as usize].prev = NULL;
        self.faces[face as usize].next = NULL;
    }

    /// The faces of the face list, in order.
    fn face_list(&self) -> Vec<u32> {
        let mut list = Vec::new();
        let mut face = self.faces[self.face_head as usize].next;
        while face != self.face_head {
            list.push(face);
            face = self.faces[face as usize].next;
        }
        list
    }

    // ---------------------------------------------------------------- allocation

    fn new_vertex(&mut self, position: Vec3) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(Vertex {
            prev: NULL,
            next: NULL,
            conflict_face: NULL,
            position,
            final_index: -1,
            reachable: false,
        });
        index
    }

    fn new_edge(&mut self) -> u32 {
        let edge = if self.edge_free != NULL {
            let edge = self.edge_free;
            self.edge_free = self.edges[edge as usize].next;
            edge
        } else {
            self.edges.push(Edge {
                prev: NULL,
                next: NULL,
                origin: NULL,
                face: NULL,
                twin: NULL,
                final_index: -1,
            });
            self.edges.len() as u32 - 1
        };
        self.edges[edge as usize].final_index = -1;
        edge
    }

    fn retire_edge(&mut self, edge: u32) {
        self.edges[edge as usize].next = self.edge_free;
        self.edge_free = edge;
    }

    fn new_face(&mut self, v1: u32, v2: u32, v3: u32) -> u32 {
        let face = if self.face_free != NULL {
            let face = self.face_free;
            self.face_free = self.faces[face as usize].next;
            face
        } else {
            let conflict_head = self.vertex_sentinel();
            self.faces.push(Face {
                prev: NULL,
                next: NULL,
                edge: NULL,
                mark: MARK_VISIBLE,
                area: 0.0,
                plane: Plane {
                    normal: Vec3::ZERO,
                    offset: 0.0,
                },
                centroid: Vec3::ZERO,
                max_conflict_distance: 0.0,
                conflict_head,
                max_conflict: NULL,
                final_index: -1,
                flipped: false,
            });
            self.faces.len() as u32 - 1
        };
        let e1 = self.new_edge();
        let e2 = self.new_edge();
        let e3 = self.new_edge();

        let (p1, p2, p3) = (
            self.vertices[v1 as usize].position,
            self.vertices[v2 as usize].position,
            self.vertices[v3 as usize].position,
        );
        let raw = (p2 - p1).cross(p3 - p1);
        let length_squared = raw.x * raw.x + raw.y * raw.y + raw.z * raw.z;
        let (length, normal) = if length_squared > 1000.0 * f32::MIN_POSITIVE {
            let length = length_squared.sqrt();
            let s = 1.0 / length;
            (length, Vec3::new(s * raw.x, s * raw.y, s * raw.z))
        } else {
            (0.0, Vec3::ZERO)
        };
        let plane = Plane {
            normal,
            offset: normal.dot(p1),
        };
        let conflict_head = self.faces[face as usize].conflict_head;
        let flipped = plane.separation(self.interior_point) > 0.0;
        self.faces[face as usize] = Face {
            prev: NULL,
            next: NULL,
            edge: e1,
            mark: MARK_VISIBLE,
            area: 0.5 * length,
            plane,
            centroid: (p1 + (p2 + p3)) * (1.0 / 3.0),
            max_conflict_distance: 0.0,
            conflict_head,
            max_conflict: NULL,
            final_index: -1,
            flipped,
        };
        let head = &mut self.vertices[conflict_head as usize];
        head.prev = conflict_head;
        head.next = conflict_head;

        for (edge, prev, next, origin) in [(e1, e3, e2, v1), (e2, e1, e3, v2), (e3, e2, e1, v3)] {
            self.edges[edge as usize] = Edge {
                prev,
                next,
                origin,
                face,
                twin: NULL,
                final_index: -1,
            };
        }
        face
    }

    fn retire_face(&mut self, face: u32) {
        // A cone face can be merged before it joins the list.
        if self.f_contains(face) {
            self.f_remove(face);
        }
        self.faces[face as usize].edge = NULL;
        self.faces[face as usize].next = self.face_free;
        self.face_free = face;
    }

    // ---------------------------------------------------------------- geometry

    fn edge_next(&self, edge: u32) -> u32 {
        self.edges[edge as usize].next
    }

    fn twin(&self, edge: u32) -> u32 {
        self.edges[edge as usize].twin
    }

    fn face_of(&self, edge: u32) -> u32 {
        self.edges[edge as usize].face
    }

    fn origin_position(&self, edge: u32) -> Vec3 {
        self.vertices[self.edges[edge as usize].origin as usize].position
    }

    /// Box3D `b3IsEdgeConvex`.
    fn is_edge_convex(&self, edge: u32, tolerance: f32) -> bool {
        let face = &self.faces[self.face_of(edge) as usize];
        let other = &self.faces[self.face_of(self.twin(edge)) as usize];
        face.plane.separation(other.centroid) < -tolerance
    }

    /// Box3D `b3IsEdgeConcave`.
    fn is_edge_concave(&self, edge: u32, tolerance: f32) -> bool {
        let face = &self.faces[self.face_of(edge) as usize];
        let other = &self.faces[self.face_of(self.twin(edge)) as usize];
        face.plane.separation(other.centroid) > tolerance
    }

    fn vertex_count_of_face(&self, face: u32) -> usize {
        let start = self.faces[face as usize].edge;
        let mut count = 0;
        let mut edge = start;
        loop {
            count += 1;
            edge = self.edge_next(edge);
            if edge == start {
                return count;
            }
        }
    }

    /// Box3D `b3LinkFace`: edge `index` of `face` and `twin` become twins.
    fn link_face(&mut self, face: u32, index: usize, twin: u32) {
        let mut edge = self.faces[face as usize].edge;
        for _ in 0..index {
            edge = self.edge_next(edge);
        }
        self.edges[edge as usize].twin = twin;
        self.edges[twin as usize].twin = edge;
    }

    /// Box3D `b3LinkFaces`.
    fn link_faces(&mut self, face1: u32, index1: usize, face2: u32, index2: usize) {
        let mut edge1 = self.faces[face1 as usize].edge;
        for _ in 0..index1 {
            edge1 = self.edge_next(edge1);
        }
        let mut edge2 = self.faces[face2 as usize].edge;
        for _ in 0..index2 {
            edge2 = self.edge_next(edge2);
        }
        self.edges[edge1 as usize].twin = edge2;
        self.edges[edge2 as usize].twin = edge1;
    }

    /// Box3D `b3NewellPlane`: the plane, centroid and area of a polygonal face.
    fn newell_plane(&mut self, face: u32) {
        let start = self.faces[face as usize].edge;
        let origin = self.origin_position(start);
        let (mut count, mut centroid, mut normal) = (0, Vec3::ZERO, Vec3::ZERO);
        let mut edge = start;
        loop {
            let twin = self.twin(edge);
            let v1 = self.origin_position(edge) - origin;
            let v2 = self.origin_position(twin) - origin;
            count += 1;
            centroid += v1;
            normal.x += (v1.y - v2.y) * (v1.z + v2.z);
            normal.y += (v1.z - v2.z) * (v1.x + v2.x);
            normal.z += (v1.x - v2.x) * (v1.y + v2.y);
            edge = self.edge_next(edge);
            if edge == start {
                break;
            }
        }
        let centroid = centroid * (1.0 / count as f32) + origin;
        let length = normal.length();
        let normal = normal * (1.0 / length);
        let f = &mut self.faces[face as usize];
        f.centroid = centroid;
        f.plane = Plane::from_normal_and_point(normal, centroid);
        f.area = 0.5 * length;
    }

    // ---------------------------------------------------------------- build

    /// Box3D `b3HullBuilder_ComputeTolerance`.
    fn compute_tolerance(&mut self, points: &[Vec3]) {
        let (lower, upper) = points.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(-f32::MAX)),
            |(lo, hi), &p| (lo.min(p), hi.max(p)),
        );
        let max_abs = lower.abs().max(upper.abs());
        let max_sum = max_abs.x + max_abs.y + max_abs.z;
        let max_coord = max_abs.x.max(max_abs.y.max(max_abs.z));
        let max_distance = (1.732_050_8 * max_coord).min(max_sum);
        self.tolerance = (3.0 * max_distance * 1.01 + max_coord) * f32::EPSILON;
        self.min_radius = 4.0 * self.tolerance;
        self.min_outside = 2.0 * self.min_radius;
    }

    /// Box3D `b3HullBuilder_BuildInitialHull`: a tetrahedron of extreme points, and every
    /// other point outside it in the conflict list of the face it is farthest outside.
    fn build_initial_hull(&mut self, points: &[Vec3]) -> bool {
        let Some((index1, index2)) = farthest_along_axes(self.tolerance, points) else {
            return false;
        };
        let Some(index3) = farthest_from_line(index1, index2, self.tolerance, points) else {
            return false;
        };
        let Some(index4) = farthest_from_plane(index1, index2, index3, self.tolerance, points)
        else {
            return false;
        };
        let (mut index2, mut index3) = (index2, index3);
        let v1 = points[index1] - points[index4];
        let v2 = points[index2] - points[index4];
        let v3 = points[index3] - points[index4];
        if v1.dot(v2.cross(v3)) < 0.0 {
            std::mem::swap(&mut index2, &mut index3);
        }
        self.interior_point =
            (Vec3::ZERO + points[index1] + points[index2] + points[index3] + points[index4]) * 0.25;

        let vertices = [index1, index2, index3, index4].map(|i| {
            let v = self.new_vertex(points[i]);
            self.v_push_back(self.vertex_head, v);
            v
        });
        let [vertex1, vertex2, vertex3, vertex4] = vertices;
        let face1 = self.new_face(vertex1, vertex2, vertex3);
        self.f_push_back(face1);
        let face2 = self.new_face(vertex4, vertex2, vertex1);
        self.f_push_back(face2);
        let face3 = self.new_face(vertex4, vertex3, vertex2);
        self.f_push_back(face3);
        let face4 = self.new_face(vertex4, vertex1, vertex3);
        self.f_push_back(face4);

        self.link_faces(face1, 0, face2, 1);
        self.link_faces(face1, 1, face3, 1);
        self.link_faces(face1, 2, face4, 1);
        self.link_faces(face2, 0, face3, 2);
        self.link_faces(face3, 0, face4, 2);
        self.link_faces(face4, 0, face2, 2);

        let faces = [face1, face2, face3, face4];
        for (index, &point) in points.iter().enumerate() {
            if index == index1 || index == index2 || index == index3 || index == index4 {
                continue;
            }
            let mut max_distance = self.min_outside;
            let mut max_face = NULL;
            for &face in &faces {
                let distance = self.faces[face as usize].plane.separation(point);
                if distance > max_distance {
                    max_distance = distance;
                    max_face = face;
                }
            }
            if max_face != NULL {
                let vertex = self.new_vertex(point);
                self.add_conflict(max_face, vertex, max_distance);
            }
        }
        true
    }

    /// Puts `vertex` at the end of `face`'s conflict list, `distance` outside it.
    fn add_conflict(&mut self, face: u32, vertex: u32, distance: f32) {
        self.vertices[vertex as usize].conflict_face = face;
        let head = self.faces[face as usize].conflict_head;
        self.v_push_back(head, vertex);
        let f = &mut self.faces[face as usize];
        if distance > f.max_conflict_distance {
            f.max_conflict_distance = distance;
            f.max_conflict = vertex;
        }
    }

    /// The vertices of a list, in order.
    fn vertex_list(&self, head: u32) -> Vec<u32> {
        let mut list = Vec::new();
        let mut node = self.vertices[head as usize].next;
        while node != head {
            list.push(node);
            node = self.vertices[node as usize].next;
        }
        list
    }

    /// Box3D `b3HullBuilder_RecacheConflicts`.
    fn recache_conflicts(&mut self, face: u32) {
        let mut max_vertex = NULL;
        let mut max_distance = self.min_outside;
        let plane = self.faces[face as usize].plane;
        for vertex in self.vertex_list(self.faces[face as usize].conflict_head) {
            let distance = plane.separation(self.vertices[vertex as usize].position);
            if distance > max_distance {
                max_distance = distance;
                max_vertex = vertex;
            }
        }
        let f = &mut self.faces[face as usize];
        f.max_conflict = max_vertex;
        f.max_conflict_distance = max_distance;
    }

    /// Box3D `b3HullBuilder_NextConflictVertex`: the point farthest outside any face.
    fn next_conflict_vertex(&self) -> u32 {
        let mut max_vertex = NULL;
        let mut max_distance = self.min_outside;
        for face in self.face_list() {
            let f = &self.faces[face as usize];
            if f.max_conflict != NULL && f.max_conflict_distance > max_distance {
                max_distance = f.max_conflict_distance;
                max_vertex = f.max_conflict;
            }
        }
        max_vertex
    }

    /// Box3D `b3HullBuilder_DrainConflictList`.
    fn drain_conflict_list(&mut self, face: u32) {
        let head = self.faces[face as usize].conflict_head;
        for orphan in self.vertex_list(head) {
            self.vertices[orphan as usize].conflict_face = NULL;
            self.v_remove(orphan);
            self.v_push_back(self.orphaned_head, orphan);
        }
    }

    /// Box3D `b3HullBuilder_EnterHorizonFace`.
    fn enter_horizon_face(&mut self, face: u32, entry_edge: u32) -> HorizonFrame {
        self.faces[face as usize].mark = MARK_DELETE;
        self.drain_conflict_list(face);
        if entry_edge != NULL {
            HorizonFrame {
                start_edge: entry_edge,
                edge: self.edge_next(entry_edge),
                started: false,
            }
        } else {
            let edge = self.faces[face as usize].edge;
            HorizonFrame {
                start_edge: edge,
                edge,
                started: false,
            }
        }
    }

    /// Box3D `b3HullBuilder_BuildHorizon`: a depth-first walk over the faces `apex` sees,
    /// collecting the edges of the horizon in order.
    fn build_horizon(&mut self, apex: u32, seed: u32) {
        self.horizon_stack.clear();
        let frame = self.enter_horizon_face(seed, NULL);
        self.horizon_stack.push(frame);
        let apex_position = self.vertices[apex as usize].position;
        while let Some(top) = self.horizon_stack.last_mut() {
            if top.started && top.edge == top.start_edge {
                self.horizon_stack.pop();
                continue;
            }
            top.started = true;
            let edge = top.edge;
            let twin = self.edges[edge as usize].twin;
            top.edge = self.edges[edge as usize].next;
            let twin_face = self.edges[twin as usize].face;
            if self.faces[twin_face as usize].mark != MARK_VISIBLE {
                continue;
            }
            let distance = self.faces[twin_face as usize]
                .plane
                .separation(apex_position);
            if distance > self.min_radius {
                let frame = self.enter_horizon_face(twin_face, twin);
                self.horizon_stack.push(frame);
            } else {
                self.horizon.push(edge);
            }
        }
    }

    /// Box3D `b3HullBuilder_BuildCone`.
    fn build_cone(&mut self, apex: u32) {
        for i in 0..self.horizon.len() {
            let edge = self.horizon[i];
            let twin = self.twin(edge);
            let face = self.new_face(
                apex,
                self.edges[edge as usize].origin,
                self.edges[twin as usize].origin,
            );
            self.cone.push(face);
            self.link_face(face, 1, twin);
        }
        let mut face1 = *self.cone.last().unwrap();
        for i in 0..self.cone.len() {
            let face2 = self.cone[i];
            self.link_faces(face1, 2, face2, 0);
            face1 = face2;
        }
    }

    /// Box3D `b3HullBuilder_DestroyEdges`: retires the ring range [begin, end).
    fn destroy_edges(&mut self, begin: u32, end: u32) {
        let mut edge = begin;
        while edge != end {
            let next = self.edge_next(edge);
            self.retire_edge(edge);
            edge = next;
        }
    }

    /// Box3D `b3HullBuilder_ConnectEdges`.
    fn connect_edges(&mut self, prev: u32, next: u32) {
        let prev_twin_face = self.face_of(self.twin(prev));
        let next_twin_face = self.face_of(self.twin(next));
        if prev_twin_face == next_twin_face {
            // `next` is redundant.
            let next_face = self.face_of(next);
            if self.faces[next_face as usize].edge == next {
                self.faces[next_face as usize].edge = prev;
            }
            let twin;
            if self.vertex_count_of_face(prev_twin_face) == 3 {
                let dead0 = self.twin(prev);
                let dead1 = self.twin(next);
                let dead2 = self.edges[dead1 as usize].prev;
                twin = self.twin(dead2);
                let opposing = prev_twin_face;
                self.faces[opposing as usize].mark = MARK_DELETE;
                self.merged_faces.push(opposing);

                let after = self.edge_next(next);
                self.edges[prev as usize].next = after;
                self.edges[after as usize].prev = prev;
                self.edges[prev as usize].twin = twin;
                self.edges[twin as usize].twin = prev;

                // Drop the redundant vertex.
                self.v_remove(self.edges[next as usize].origin);

                self.retire_edge(dead0);
                self.retire_edge(dead1);
                self.retire_edge(dead2);
            } else {
                twin = self.twin(next);
                let twin_face = self.face_of(twin);
                let prev_twin = self.twin(prev);
                if self.faces[twin_face as usize].edge == prev_twin {
                    self.faces[twin_face as usize].edge = twin;
                }
                let after_prev_twin = self.edge_next(prev_twin);
                self.edges[twin as usize].next = after_prev_twin;
                self.edges[after_prev_twin as usize].prev = twin;
                self.retire_edge(prev_twin);

                let after = self.edge_next(next);
                self.edges[prev as usize].next = after;
                self.edges[after as usize].prev = prev;
                self.edges[prev as usize].twin = twin;
                self.edges[twin as usize].twin = prev;

                self.v_remove(self.edges[next as usize].origin);
            }
            let twin_face = self.face_of(twin);
            self.newell_plane(twin_face);
            self.recache_conflicts(twin_face);
        } else {
            self.edges[prev as usize].next = next;
            self.edges[next as usize].prev = prev;
        }
    }

    /// Box3D `b3HullBuilder_AbsorbFaces`: the merged faces' conflicts go to `face` or the
    /// orphans, and the merged faces are retired.
    fn absorb_faces(&mut self, face: u32) {
        for i in 0..self.merged_faces.len() {
            let merged = self.merged_faces[i];
            let head = self.faces[merged as usize].conflict_head;
            for vertex in self.vertex_list(head) {
                self.v_remove(vertex);
                let position = self.vertices[vertex as usize].position;
                let distance = self.faces[face as usize].plane.separation(position);
                if distance > self.min_outside {
                    self.add_conflict(face, vertex, distance);
                } else {
                    self.v_push_back(self.orphaned_head, vertex);
                    self.vertices[vertex as usize].conflict_face = NULL;
                }
            }
            self.retire_face(merged);
        }
    }

    /// Box3D `b3HullBuilder_ConnectFaces`: merges the face across `edge` into `edge`'s face.
    fn connect_faces(&mut self, edge: u32) {
        let face = self.face_of(edge);
        let twin = self.twin(edge);
        let twin_face = self.face_of(twin);
        let mut edge_prev = self.edges[edge as usize].prev;
        let mut edge_next = self.edge_next(edge);
        let mut twin_prev = self.edges[twin as usize].prev;
        let mut twin_next = self.edge_next(twin);

        while self.face_of(self.twin(edge_prev)) == twin_face {
            edge_prev = self.edges[edge_prev as usize].prev;
            twin_next = self.edge_next(twin_next);
        }
        while self.face_of(self.twin(edge_next)) == twin_face {
            edge_next = self.edge_next(edge_next);
            twin_prev = self.edges[twin_prev as usize].prev;
        }

        self.faces[face as usize].edge = edge_prev;

        // Discard the opposing face.
        self.merged_faces.clear();
        self.merged_faces.push(twin_face);
        self.faces[twin_face as usize].mark = MARK_DELETE;
        self.faces[twin_face as usize].edge = NULL;

        let end = self.edge_next(twin_prev);
        let mut absorbed = twin_next;
        while absorbed != end {
            self.edges[absorbed as usize].face = face;
            absorbed = self.edge_next(absorbed);
        }

        let after_edge_prev = self.edge_next(edge_prev);
        self.destroy_edges(after_edge_prev, edge_next);
        let after_twin_prev = self.edge_next(twin_prev);
        self.destroy_edges(after_twin_prev, twin_next);

        self.connect_edges(edge_prev, twin_next);
        self.connect_edges(twin_prev, edge_next);

        self.newell_plane(face);
        self.recache_conflicts(face);
        self.absorb_faces(face);
    }

    /// Box3D `b3HullBuilder_MergeConcave`.
    fn merge_concave(&mut self, face: u32) -> bool {
        let start = self.faces[face as usize].edge;
        let mut edge = start;
        loop {
            let twin = self.twin(edge);
            if self.is_edge_concave(edge, self.min_radius)
                || self.is_edge_concave(twin, self.min_radius)
            {
                self.connect_faces(edge);
                return true;
            }
            edge = self.edge_next(edge);
            if edge == start {
                return false;
            }
        }
    }

    /// Box3D `b3HullBuilder_MergeCoplanar`.
    fn merge_coplanar(&mut self, face: u32) -> bool {
        let start = self.faces[face as usize].edge;
        let mut edge = start;
        loop {
            let twin = self.twin(edge);
            if !self.is_edge_convex(edge, self.min_radius)
                || !self.is_edge_convex(twin, self.min_radius)
            {
                self.connect_faces(edge);
                return true;
            }
            edge = self.edge_next(edge);
            if edge == start {
                return false;
            }
        }
    }

    /// Box3D `b3HullBuilder_MergeFaces`: flipped cone faces into their largest neighbour, then
    /// concave and coplanar ones.
    fn merge_faces(&mut self) {
        for i in 0..self.cone.len() {
            let face = self.cone[i];
            if self.faces[face as usize].mark == MARK_VISIBLE && self.faces[face as usize].flipped {
                self.faces[face as usize].flipped = false;
                let mut best_area = 0.0;
                let mut best_edge = NULL;
                let start = self.faces[face as usize].edge;
                let mut edge = start;
                loop {
                    let area = self.faces[self.face_of(self.twin(edge)) as usize].area;
                    if area > best_area {
                        best_area = area;
                        best_edge = edge;
                    }
                    edge = self.edge_next(edge);
                    if edge == start {
                        break;
                    }
                }
                self.connect_faces(best_edge);
            }
        }
        for i in 0..self.cone.len() {
            let face = self.cone[i];
            if self.faces[face as usize].mark == MARK_VISIBLE {
                while self.merge_concave(face) {}
            }
        }
        for i in 0..self.cone.len() {
            let face = self.cone[i];
            if self.faces[face as usize].mark == MARK_VISIBLE {
                while self.merge_coplanar(face) {}
            }
        }
    }

    /// Box3D `b3HullBuilder_ResolveVertices`: each orphan goes to the cone face it is farthest
    /// outside, or is dropped as interior.
    fn resolve_vertices(&mut self) {
        for vertex in self.vertex_list(self.orphaned_head) {
            self.v_remove(vertex);
            let position = self.vertices[vertex as usize].position;
            let mut max_distance = self.min_outside;
            let mut max_face = NULL;
            for i in 0..self.cone.len() {
                let face = self.cone[i];
                if self.faces[face as usize].mark == MARK_VISIBLE {
                    let distance = self.faces[face as usize].plane.separation(position);
                    if distance > max_distance {
                        max_distance = distance;
                        max_face = face;
                    }
                }
            }
            if max_face != NULL {
                self.add_conflict(max_face, vertex, max_distance);
            }
        }
        debug_assert!(self.v_empty(self.orphaned_head));
    }

    /// Box3D `b3HullBuilder_ResolveFaces`: the deleted faces leave the list, the cone joins it.
    fn resolve_faces(&mut self) {
        for face in self.face_list() {
            if self.faces[face as usize].mark == MARK_DELETE && self.f_contains(face) {
                let start = self.faces[face as usize].edge;
                let mut edge = start;
                loop {
                    let next = self.edge_next(edge);
                    self.retire_edge(edge);
                    edge = next;
                    if edge == start {
                        break;
                    }
                }
                self.retire_face(face);
            }
        }
        for i in 0..self.cone.len() {
            let face = self.cone[i];
            if self.faces[face as usize].mark != MARK_DELETE {
                self.f_push_back(face);
            }
        }
    }

    /// Box3D `b3HullBuilder_AddVertexToHull`.
    fn add_vertex_to_hull(&mut self, vertex: u32) {
        let face = self.vertices[vertex as usize].conflict_face;
        self.vertices[vertex as usize].conflict_face = NULL;
        self.v_remove(vertex);
        self.v_push_back(self.vertex_head, vertex);
        self.horizon.clear();
        self.build_horizon(vertex, face);
        self.cone.clear();
        self.build_cone(vertex);
        self.merge_faces();
        self.resolve_vertices();
        self.resolve_faces();
    }

    /// Box3D `b3HullBuilder_CleanHull`: drops unreachable vertices, counts, and moves back by
    /// `origin`.
    fn clean_hull(&mut self, origin: Vec3) {
        let (mut face_count, mut half_edge_count) = (0, 0);
        for face in self.face_list() {
            let start = self.faces[face as usize].edge;
            let mut edge = start;
            loop {
                let o = self.edges[edge as usize].origin;
                self.vertices[o as usize].reachable = true;
                edge = self.edge_next(edge);
                half_edge_count += 1;
                if edge == start {
                    break;
                }
            }
            let f = &mut self.faces[face as usize];
            f.plane.offset += f.plane.normal.dot(origin);
            f.centroid += origin;
            face_count += 1;
        }
        let mut vertex_count = 0;
        for vertex in self.vertex_list(self.vertex_head) {
            if self.vertices[vertex as usize].reachable {
                self.vertices[vertex as usize].position += origin;
                vertex_count += 1;
            } else {
                self.v_remove(vertex);
            }
        }
        self.interior_point += origin;
        self.final_vertex_count = vertex_count;
        self.final_half_edge_count = half_edge_count;
        self.final_face_count = face_count;
    }
}

/// Box3D `b3FindFarthestPointsAlongCardinalAxes`.
fn farthest_along_axes(tolerance: f32, points: &[Vec3]) -> Option<(usize, usize)> {
    let v0 = points[0];
    let (mut min_pt, mut max_pt) = ([v0; 3], [v0; 3]);
    let (mut min_index, mut max_index) = ([0usize; 3], [0usize; 3]);
    for (i, &v) in points.iter().enumerate().skip(1) {
        for axis in 0..3 {
            if v[axis] < min_pt[axis][axis] {
                min_pt[axis] = v;
                min_index[axis] = i;
            } else if v[axis] > max_pt[axis][axis] {
                max_pt[axis] = v;
                max_index[axis] = i;
            }
        }
    }
    let distance = Vec3::new(
        max_pt[0].x - min_pt[0].x,
        max_pt[1].y - min_pt[1].y,
        max_pt[2].z - min_pt[2].z,
    );
    // Box3D `b3MaxElementIndex`.
    let axis = if distance.x < distance.y {
        if distance.y < distance.z {
            2
        } else {
            1
        }
    } else if distance.x < distance.z {
        2
    } else {
        0
    };
    (distance[axis] > 2.0 * tolerance).then_some((min_index[axis], max_index[axis]))
}

/// Box3D `b3FindFarthestPointFromLine`.
fn farthest_from_line(
    index1: usize,
    index2: usize,
    tolerance: f32,
    points: &[Vec3],
) -> Option<usize> {
    let (a, b) = (points[index1], points[index2]);
    let ab = b - a;
    let inv_length_squared = 1.0 / ab.dot(ab);
    let mut max_distance_squared = 4.0 * tolerance * tolerance;
    let mut max_index = None;
    for (i, &p) in points.iter().enumerate() {
        if i == index1 || i == index2 {
            continue;
        }
        let cross = (p - a).cross(ab);
        let distance_squared = cross.dot(cross) * inv_length_squared;
        if distance_squared > max_distance_squared {
            max_distance_squared = distance_squared;
            max_index = Some(i);
        }
    }
    max_index
}

/// Box3D `b3FindFarthestPointFromPlane`.
fn farthest_from_plane(
    index1: usize,
    index2: usize,
    index3: usize,
    tolerance: f32,
    points: &[Vec3],
) -> Option<usize> {
    let plane = Plane::from_points(points[index1], points[index2], points[index3]);
    let mut max_distance = 2.0 * tolerance;
    let mut max_index = None;
    for (i, &p) in points.iter().enumerate() {
        if i == index1 || i == index2 || i == index3 {
            continue;
        }
        let distance = plane.separation(p).abs();
        if distance > max_distance {
            max_distance = distance;
            max_index = Some(i);
        }
    }
    max_index
}

/// Box3D `b3CreateHull`'s construction: the hull of `points` with at most `max_vertices`
/// vertices (4 to 128), numbered as Box3D numbers them, or `None` for degenerate input.
pub(crate) fn build(points: &[Vec3], max_vertices: usize) -> Option<BuiltHull> {
    if points.len() < 4 {
        return None;
    }
    let max_vertices = max_vertices.clamp(4, crate::hull::MAX_HULL_VERTICES);
    let origin = points[0];
    let shifted: Vec<Vec3> = points.iter().map(|&p| p - origin).collect();

    let mut b = Builder::new();
    b.compute_tolerance(&shifted);
    if !b.build_initial_hull(&shifted) {
        return None;
    }
    let mut budget = (max_vertices as i64 - 4).clamp(0, MAX_COUNT as i64 - 4);
    let mut vertex = b.next_conflict_vertex();
    while vertex != NULL && budget > 0 {
        b.add_vertex_to_hull(vertex);
        vertex = b.next_conflict_vertex();
        budget -= 1;
    }
    b.clean_hull(origin);

    let (v, e, f) = (
        b.final_vertex_count,
        b.final_half_edge_count / 2,
        b.final_face_count,
    );
    if !(v as i64 - e as i64 + f as i64 == 2 && f >= 4) {
        return None;
    }
    if v > crate::hull::MAX_HULL_VERTICES
        || f > crate::hull::MAX_HULL_FACES
        || b.final_half_edge_count > 2 * crate::hull::MAX_HULL_EDGES
    {
        return None;
    }

    // Number the vertices in list order, the faces in list order and the half-edges in twin
    // pairs as each face's ring meets them.
    let vertex_order = b.vertex_list(b.vertex_head);
    for (i, &vertex) in vertex_order.iter().enumerate() {
        b.vertices[vertex as usize].final_index = i as i32;
    }
    let face_order = b.face_list();
    let mut edge_order = Vec::new();
    for (i, &face) in face_order.iter().enumerate() {
        b.faces[face as usize].final_index = i as i32;
        let start = b.faces[face as usize].edge;
        let mut edge = start;
        loop {
            if b.edges[edge as usize].final_index < 0 {
                let twin = b.edges[edge as usize].twin;
                b.edges[edge as usize].final_index = edge_order.len() as i32;
                edge_order.push(edge);
                b.edges[twin as usize].final_index = edge_order.len() as i32;
                edge_order.push(twin);
            }
            edge = b.edges[edge as usize].next;
            if edge == start {
                break;
            }
        }
    }

    let points: Vec<Vec3> = vertex_order
        .iter()
        .map(|&v| b.vertices[v as usize].position)
        .collect();
    let mut vertex_edges = vec![0u8; points.len()];
    let edges: Vec<HalfEdge> = edge_order
        .iter()
        .enumerate()
        .map(|(index, &edge)| {
            let e = b.edges[edge as usize];
            let origin = b.vertices[e.origin as usize].final_index as u8;
            vertex_edges[origin as usize] = index as u8;
            HalfEdge {
                next: b.edges[e.next as usize].final_index as u8,
                twin: b.edges[e.twin as usize].final_index as u8,
                origin,
                face: b.faces[e.face as usize].final_index as u8,
            }
        })
        .collect();
    let face_edges = face_order
        .iter()
        .map(|&face| b.edges[b.faces[face as usize].edge as usize].final_index as u8)
        .collect();
    let planes = face_order
        .iter()
        .map(|&face| b.faces[face as usize].plane)
        .collect();
    Some(BuiltHull {
        points,
        planes,
        edges,
        face_edges,
        vertex_edges,
    })
}
