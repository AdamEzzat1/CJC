//! Integration tests for loop analysis — end-to-end from CJC source code.

use cjc_mir::cfg::CfgBuilder;
use cjc_mir::dominators::DominatorTree;
use cjc_mir::loop_analysis::{compute_loop_tree, LoopId};

/// Parse CJC source → MIR, build CFG for first user function, return loop tree.
fn loop_tree_from_source(src: &str) -> cjc_mir::loop_analysis::LoopTree {
    let (program, diags) = cjc_parser::parse_source(src);
    if diags.has_errors() {
        panic!("CJC parse failed for:\n{}\n", src);
    }
    let mir = cjc_mir_exec::lower_to_mir(&program);
    let func = mir
        .functions
        .iter()
        .find(|f| f.name != "__main")
        .unwrap_or(&mir.functions[0]);
    let cfg = CfgBuilder::build(&func.body);
    let domtree = DominatorTree::compute(&cfg);
    compute_loop_tree(&cfg, &domtree)
}

#[test]
fn test_source_simple_while_loop() {
    let src = r#"
fn count(n: i64) -> i64 {
    let mut i: i64 = 0;
    while i < n {
        i = i + 1;
    }
    return i;
}
print(count(10));
"#;
    let tree = loop_tree_from_source(src);
    assert_eq!(tree.len(), 1, "should detect 1 loop");
    let loop0 = tree.get(LoopId(0));
    assert_eq!(loop0.depth, 0);
    assert!(loop0.parent.is_none());
    assert!(!loop0.body_blocks.is_empty());
    assert!(!loop0.exit_blocks.is_empty());
}

#[test]
fn test_source_nested_while_loops() {
    let src = r#"
fn matrix_sum(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            total = total + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
print(matrix_sum(3));
"#;
    let tree = loop_tree_from_source(src);
    assert_eq!(tree.len(), 2, "should detect 2 loops (outer + inner)");
    let outer = tree.loops.iter().find(|l| l.depth == 0).unwrap();
    let inner = tree.loops.iter().find(|l| l.depth == 1).unwrap();
    assert!(outer.children.contains(&inner.id));
    assert_eq!(inner.parent, Some(outer.id));
    assert!(tree.is_nested_in(inner.id, outer.id));
    assert_eq!(tree.max_depth(), 1);
}

#[test]
fn test_source_sequential_loops() {
    let src = r#"
fn two_loops(n: i64) -> i64 {
    let mut a: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        a = a + 1;
        i = i + 1;
    }
    let mut b: i64 = 0;
    let mut j: i64 = 0;
    while j < n {
        b = b + 2;
        j = j + 1;
    }
    return a + b;
}
print(two_loops(3));
"#;
    let tree = loop_tree_from_source(src);
    assert_eq!(tree.len(), 2, "should detect 2 independent loops");
    assert!(tree.loops.iter().all(|l| l.depth == 0));
    assert_eq!(tree.root_loops().len(), 2);
}

#[test]
fn test_source_no_loops() {
    let src = r#"
fn add(a: i64, b: i64) -> i64 { a + b }
print(add(1, 2));
"#;
    let tree = loop_tree_from_source(src);
    assert!(tree.is_empty());
}

#[test]
fn test_source_loop_with_branch() {
    let src = r#"
fn abs_sum(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        if i > 5 {
            total = total + i;
        } else {
            total = total - i;
        }
        i = i + 1;
    }
    return total;
}
print(abs_sum(10));
"#;
    let tree = loop_tree_from_source(src);
    assert_eq!(tree.len(), 1, "if/else inside loop should not create extra loop");
}

#[test]
fn test_source_loop_tree_determinism() {
    let src = r#"
fn nested(n: i64) -> i64 {
    let mut s: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < i {
            s = s + j;
            j = j + 1;
        }
        i = i + 1;
    }
    return s;
}
print(nested(5));
"#;
    let tree1 = loop_tree_from_source(src);
    let tree2 = loop_tree_from_source(src);
    assert_eq!(tree1.len(), tree2.len());
    for i in 0..tree1.len() {
        assert_eq!(tree1.loops[i].header, tree2.loops[i].header);
        assert_eq!(tree1.loops[i].body_blocks, tree2.loops[i].body_blocks);
        assert_eq!(tree1.loops[i].depth, tree2.loops[i].depth);
        assert_eq!(tree1.loops[i].parent, tree2.loops[i].parent);
    }
}

#[test]
fn test_source_triple_nested() {
    let src = r#"
fn cube(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            let mut k: i64 = 0;
            while k < n {
                total = total + 1;
                k = k + 1;
            }
            j = j + 1;
        }
        i = i + 1;
    }
    return total;
}
print(cube(3));
"#;
    let tree = loop_tree_from_source(src);
    assert_eq!(tree.len(), 3, "should detect 3 loops");
    assert_eq!(tree.max_depth(), 2);
    let depths: Vec<u32> = tree.loops.iter().map(|l| l.depth).collect();
    assert!(depths.contains(&0));
    assert!(depths.contains(&1));
    assert!(depths.contains(&2));
    assert_eq!(tree.root_loops().len(), 1);
}
