//! Phase C test C5: Map Completion & Set Type

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

#[test]
fn c5_map_insert_get() {
    let out = run_mir(r#"
let m = Map.new();
m.insert("hello", 42);
let val = m.get("hello");
print(val);
"#);
    assert_eq!(out, vec!["42"]);
}

#[test]
fn c5_map_remove() {
    let out = run_mir(r#"
let m = Map.new();
m.insert("a", 1);
m.insert("b", 2);
m.remove("a");
print(m.len());
print(m.get("a"));
"#);
    assert_eq!(out[0], "1");
    assert_eq!(out[1], "void");
}

#[test]
fn c5_map_contains_key() {
    let out = run_mir(r#"
let m = Map.new();
m.insert("x", 10);
print(m.contains_key("x"));
print(m.contains_key("y"));
"#);
    assert_eq!(out[0], "true");
    assert_eq!(out[1], "false");
}

#[test]
fn c5_map_keys_values() {
    let out = run_mir(r#"
let m = Map.new();
m.insert("a", 1);
m.insert("b", 2);
print(m.len());
"#);
    assert_eq!(out[0], "2");
}

#[test]
fn c5_map_overwrite() {
    let out = run_mir(r#"
let m = Map.new();
m.insert("key", 10);
m.insert("key", 20);
print(m.get("key"));
print(m.len());
"#);
    assert_eq!(out[0], "20");
    assert_eq!(out[1], "1");
}

#[test]
fn c5_set_add_contains() {
    let out = run_mir(r#"
let s = Set.new();
s.add(1);
s.add(2);
s.add(3);
print(s.contains(2));
print(s.contains(5));
"#);
    assert_eq!(out[0], "true");
    assert_eq!(out[1], "false");
}

#[test]
fn c5_set_remove() {
    let out = run_mir(r#"
let s = Set.new();
s.add(10);
s.add(20);
s.remove(10);
print(s.len());
print(s.contains(10));
"#);
    assert_eq!(out[0], "1");
    assert_eq!(out[1], "false");
}

#[test]
fn c5_set_dedup() {
    let out = run_mir(r#"
let s = Set.new();
s.add(1);
s.add(1);
s.add(1);
print(s.len());
"#);
    assert_eq!(out[0], "1");
}

#[test]
fn c5_set_len() {
    let out = run_mir(r#"
let s = Set.new();
print(s.len());
s.add("a");
s.add("b");
s.add("c");
print(s.len());
"#);
    assert_eq!(out[0], "0");
    assert_eq!(out[1], "3");
}

#[test]
fn c5_determinism() {
    let src = r#"
let m = Map.new();
m.insert("x", 1);
m.insert("y", 2);
m.insert("z", 3);
print(m.get("x"));
print(m.get("y"));
print(m.get("z"));
print(m.len());
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}

#[test]
fn c5_map_int_keys() {
    let out = run_mir(r#"
let m = Map.new();
m.insert(0, "first");
m.insert(1, "second");
print(m.get(0));
print(m.get(1));
"#);
    assert_eq!(out[0], "first");
    assert_eq!(out[1], "second");
}

#[test]
fn c5_set_to_array() {
    let out = run_mir(r#"
let s = Set.new();
s.add(10);
s.add(20);
let arr = s.to_array();
print(len(arr));
"#);
    assert_eq!(out[0], "2");
}
