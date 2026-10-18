//! Integration tests for the scene hierarchy on `ChildOf` (parent and ordered children, reparenting, cycle prevention, despawn recursive).

use korppi_ecs::World;
use korppi_scene::hierarchy::{
    attach_child, despawn_recursive, detach_child, get_ancestors, get_descendants, get_root,
    HierarchyError,
};

#[test]
fn attaching_child_creates_bidirectional_linkage() {
    let mut world = World::new();

    let parent = world.spawn(());
    let child1 = world.spawn(());
    let child2 = world.spawn(());

    attach_child(&mut world, parent, child1).expect("Failed to attach child1");
    attach_child(&mut world, parent, child2).expect("Failed to attach child2");

    // The parent's children are child1 and child2, in the order they were attached
    assert_eq!(world.children(parent), [child1, child2]);

    // Each child's parent is the parent
    assert_eq!(world.parent_of(child1), Some(parent));
    assert_eq!(world.parent_of(child2), Some(parent));
}

#[test]
fn reparenting_moves_child_between_parents() {
    let mut world = World::new();

    let parent_a = world.spawn(());
    let parent_b = world.spawn(());
    let child = world.spawn(());

    attach_child(&mut world, parent_a, child).unwrap();
    assert_eq!(world.children(parent_a), [child]);

    // Move child from parent A to parent B
    attach_child(&mut world, parent_b, child).unwrap();

    // Parent A no longer has the child
    assert!(world.children(parent_a).is_empty());

    // Parent B should now have child
    assert_eq!(world.children(parent_b), [child]);

    // Child's parent should be parent B
    assert_eq!(world.parent_of(child), Some(parent_b));
}

#[test]
fn detaching_child_removes_linkage() {
    let mut world = World::new();

    let parent = world.spawn(());
    let child = world.spawn(());

    attach_child(&mut world, parent, child).unwrap();
    assert_eq!(world.children(parent), [child]);
    assert_eq!(world.parent_of(child), Some(parent));

    detach_child(&mut world, parent, child).unwrap();
    assert!(world.children(parent).is_empty());
    assert_eq!(world.parent_of(child), None);
}

#[test]
fn cycle_detection_prevents_recursive_parenting() {
    let mut world = World::new();

    let node_a = world.spawn(());
    let node_b = world.spawn(());
    let node_c = world.spawn(());

    // A -> B -> C
    attach_child(&mut world, node_a, node_b).unwrap();
    attach_child(&mut world, node_b, node_c).unwrap();

    // Self parenting should fail
    let self_err = attach_child(&mut world, node_a, node_a);
    assert_eq!(self_err, Err(HierarchyError::SelfParenting(node_a)));

    // Cycle: C as parent of A should fail because A is an ancestor of C
    let cycle_err = attach_child(&mut world, node_c, node_a);
    assert_eq!(
        cycle_err,
        Err(HierarchyError::CycleDetected {
            parent: node_c,
            child: node_a
        })
    );
}

#[test]
fn hierarchy_traversal_finds_ancestors_descendants_and_root() {
    let mut world = World::new();

    let root_node = world.spawn(());
    let branch = world.spawn(());
    let leaf1 = world.spawn(());
    let leaf2 = world.spawn(());

    attach_child(&mut world, root_node, branch).unwrap();
    attach_child(&mut world, branch, leaf1).unwrap();
    attach_child(&mut world, branch, leaf2).unwrap();

    // Root check
    assert_eq!(get_root(&world, leaf1), root_node);
    assert_eq!(get_root(&world, leaf2), root_node);
    assert_eq!(get_root(&world, branch), root_node);
    assert_eq!(get_root(&world, root_node), root_node);

    // Ancestors check for leaf1: branch, root_node
    assert_eq!(get_ancestors(&world, leaf1), vec![branch, root_node]);

    // Descendants check for root_node: branch, leaf1, leaf2
    let descendants = get_descendants(&world, root_node);
    assert_eq!(descendants.len(), 3);
    assert!(descendants.contains(&branch));
    assert!(descendants.contains(&leaf1));
    assert!(descendants.contains(&leaf2));
}

#[test]
fn despawn_recursive_destroys_entire_subtree() {
    let mut world = World::new();

    let root_node = world.spawn(());
    let child1 = world.spawn(());
    let child2 = world.spawn(());
    let grandchild = world.spawn(());

    attach_child(&mut world, root_node, child1).unwrap();
    attach_child(&mut world, root_node, child2).unwrap();
    attach_child(&mut world, child1, grandchild).unwrap();

    assert_eq!(world.entity_count(), 4);

    // Despawn child1 subtree (child1 + grandchild)
    despawn_recursive(&mut world, child1);

    assert_eq!(world.entity_count(), 2);
    assert!(world.is_alive(root_node));
    assert!(world.is_alive(child2));
    assert!(!world.is_alive(child1));
    assert!(!world.is_alive(grandchild));

    // Root's children should only contain child2
    assert_eq!(world.children(root_node), [child2]);
}
