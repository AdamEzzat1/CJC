//! Phase C test C4: Sorting & Tensor Indexing

fn run_mir(src: &str) -> Vec<String> {
    let (tokens, _) = cjc_lexer::Lexer::new(src).tokenize();
    let (program, _) = cjc_parser::Parser::new(tokens).parse_program();
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    executor.output
}

fn parse_tensor_data(s: &str) -> Vec<f64> {
    let data_start = s.find("data=[").expect("no data= in tensor output") + 6;
    let data_end = s[data_start..].find(']').expect("no closing ]") + data_start;
    let data_str = &s[data_start..data_end];
    data_str.split(", ").map(|v| v.trim().parse::<f64>().unwrap()).collect()
}

#[test]
fn c4_argsort_ascending() {
    let out = run_mir(r#"
let t = Tensor.from_vec([3.0, 1.0, 2.0], [3]);
let idx = argsort(t);
print(idx);
"#);
    let data = parse_tensor_data(&out[0]);
    // [3,1,2] → sorted [1,2,3] → indices [1,2,0]
    assert_eq!(data, vec![1.0, 2.0, 0.0]);
}

#[test]
fn c4_argsort_with_ties() {
    let out = run_mir(r#"
let t = Tensor.from_vec([2.0, 2.0, 1.0], [3]);
let idx = argsort(t);
print(idx);
"#);
    let data = parse_tensor_data(&out[0]);
    // [2,2,1] → sorted [1,2,2] → indices [2,0,1] (stable? depends on sort_by)
    assert_eq!(data[0], 2.0); // index of 1.0 comes first
}

#[test]
fn c4_gather_1d() {
    let out = run_mir(r#"
let t = Tensor.from_vec([10.0, 20.0, 30.0, 40.0], [4]);
let idx = Tensor.from_vec([3.0, 0.0, 1.0], [3]);
let result = gather(t, 0, idx);
print(result);
"#);
    let data = parse_tensor_data(&out[0]);
    assert_eq!(data, vec![40.0, 10.0, 20.0]);
}

#[test]
fn c4_gather_2d() {
    // 2x3 matrix, gather dim=1 with [[2,0],[1,2]]
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
let idx = Tensor.from_vec([2.0, 0.0, 1.0, 2.0], [2, 2]);
let result = gather(t, 1, idx);
print(result);
"#);
    let data = parse_tensor_data(&out[0]);
    // row 0: t[0][2]=3, t[0][0]=1
    // row 1: t[1][1]=5, t[1][2]=6
    assert_eq!(data, vec![3.0, 1.0, 5.0, 6.0]);
}

#[test]
fn c4_scatter_1d() {
    let out = run_mir(r#"
let t = Tensor.zeros([4]);
let idx = Tensor.from_vec([1.0, 3.0], [2]);
let src = Tensor.from_vec([10.0, 20.0], [2]);
let result = scatter(t, 0, idx, src);
print(result);
"#);
    let data = parse_tensor_data(&out[0]);
    // result = [0, 10, 0, 20]
    assert_eq!(data, vec![0.0, 10.0, 0.0, 20.0]);
}

#[test]
fn c4_index_select_rows() {
    // Select rows 0 and 2 from a 3x2 matrix
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
let idx = Tensor.from_vec([0.0, 2.0], [2]);
let result = index_select(t, 0, idx);
print(result);
"#);
    let data = parse_tensor_data(&out[0]);
    // rows [0,2] = [[1,2],[5,6]]
    assert_eq!(data, vec![1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn c4_index_select_cols() {
    // Select columns 1 and 0 from a 2x3 matrix
    let out = run_mir(r#"
let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
let idx = Tensor.from_vec([1.0, 0.0], [2]);
let result = index_select(t, 1, idx);
print(result);
"#);
    let data = parse_tensor_data(&out[0]);
    // cols [1,0] → [[2,1],[5,4]]
    assert_eq!(data, vec![2.0, 1.0, 5.0, 4.0]);
}

#[test]
fn c4_argsort_determinism() {
    let src = r#"
let t = Tensor.from_vec([5.0, 2.0, 8.0, 1.0, 4.0], [5]);
let idx = argsort(t);
print(idx);
"#;
    let out1 = run_mir(src);
    let out2 = run_mir(src);
    assert_eq!(out1, out2);
}

#[test]
fn c4_one_hot_exists() {
    // Verify one_hot still works (was already wired in Phase B4)
    let out = run_mir(r#"
let oh = one_hot([0, 2, 1], 3);
print(oh);
"#);
    let data = parse_tensor_data(&out[0]);
    // 3x3 identity-ish: [[1,0,0],[0,0,1],[0,1,0]]
    assert_eq!(data, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn c4_index_select_1d() {
    let out = run_mir(r#"
let t = Tensor.from_vec([10.0, 20.0, 30.0, 40.0, 50.0], [5]);
let idx = Tensor.from_vec([4.0, 2.0, 0.0], [3]);
let result = index_select(t, 0, idx);
print(result);
"#);
    let data = parse_tensor_data(&out[0]);
    assert_eq!(data, vec![50.0, 30.0, 10.0]);
}
