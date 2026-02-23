// CJC Test Suite — cjc-eval (28 tests)
// Source: crates/cjc-eval/src/lib.rs
// These tests are extracted from the inline #[cfg(test)] modules for regression tracking.

use cjc_eval::*;
use cjc_ast::*;
use cjc_runtime::{Tensor, Value};

// -- Helpers for building AST nodes -------------------------------------

fn span() -> Span { Span::dummy() }
fn ident(name: &str) -> Ident { Ident::dummy(name) }
fn int_expr(v: i64) -> Expr { Expr { kind: ExprKind::IntLit(v), span: span() } }
fn float_expr(v: f64) -> Expr { Expr { kind: ExprKind::FloatLit(v), span: span() } }
fn bool_expr(v: bool) -> Expr { Expr { kind: ExprKind::BoolLit(v), span: span() } }
fn string_expr(s: &str) -> Expr { Expr { kind: ExprKind::StringLit(s.to_string()), span: span() } }
fn ident_expr(name: &str) -> Expr { Expr { kind: ExprKind::Ident(ident(name)), span: span() } }

fn binary(op: BinOp, left: Expr, right: Expr) -> Expr {
    Expr { kind: ExprKind::Binary { op, left: Box::new(left), right: Box::new(right) }, span: span() }
}
fn unary(op: UnaryOp, operand: Expr) -> Expr {
    Expr { kind: ExprKind::Unary { op, operand: Box::new(operand) }, span: span() }
}
fn call(callee: Expr, args: Vec<Expr>) -> Expr {
    let call_args: Vec<CallArg> = args.into_iter().map(|value| CallArg { name: None, value, span: span() }).collect();
    Expr { kind: ExprKind::Call { callee: Box::new(callee), args: call_args }, span: span() }
}
fn field_expr(object: Expr, name: &str) -> Expr {
    Expr { kind: ExprKind::Field { object: Box::new(object), name: ident(name) }, span: span() }
}
fn assign_expr(target: Expr, value: Expr) -> Expr {
    Expr { kind: ExprKind::Assign { target: Box::new(target), value: Box::new(value) }, span: span() }
}
fn pipe_expr(left: Expr, right: Expr) -> Expr {
    Expr { kind: ExprKind::Pipe { left: Box::new(left), right: Box::new(right) }, span: span() }
}
fn array_expr(elems: Vec<Expr>) -> Expr { Expr { kind: ExprKind::ArrayLit(elems), span: span() } }
fn struct_lit(name: &str, fields: Vec<(&str, Expr)>) -> Expr {
    Expr { kind: ExprKind::StructLit { name: ident(name), fields: fields.into_iter().map(|(n, v)| FieldInit { name: ident(n), value: v, span: span() }).collect() }, span: span() }
}
fn index_expr(object: Expr, index: Expr) -> Expr {
    Expr { kind: ExprKind::Index { object: Box::new(object), index: Box::new(index) }, span: span() }
}
fn multi_index_expr(object: Expr, indices: Vec<Expr>) -> Expr {
    Expr { kind: ExprKind::MultiIndex { object: Box::new(object), indices }, span: span() }
}
fn let_stmt(name: &str, init: Expr) -> Stmt {
    Stmt { kind: StmtKind::Let(LetStmt { name: ident(name), mutable: false, ty: None, init: Box::new(init) }), span: span() }
}
fn let_mut_stmt(name: &str, init: Expr) -> Stmt {
    Stmt { kind: StmtKind::Let(LetStmt { name: ident(name), mutable: true, ty: None, init: Box::new(init) }), span: span() }
}
fn expr_stmt(expr: Expr) -> Stmt { Stmt { kind: StmtKind::Expr(expr), span: span() } }
fn return_stmt(expr: Option<Expr>) -> Stmt { Stmt { kind: StmtKind::Return(expr), span: span() } }
fn dummy_type_expr() -> TypeExpr { TypeExpr { kind: TypeExprKind::Named { name: ident("i64"), args: vec![] }, span: span() } }
fn make_param(name: &str) -> Param { Param { name: ident(name), ty: dummy_type_expr(), span: span() } }
fn make_fn_decl(name: &str, params: Vec<&str>, body: Block) -> Decl {
    Decl { kind: DeclKind::Fn(FnDecl { name: ident(name), type_params: vec![], params: params.into_iter().map(|n| make_param(n)).collect(), return_type: None, body, is_nogc: false }), span: span() }
}
fn make_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Block { Block { stmts, expr: expr.map(Box::new), span: span() } }
fn make_struct_decl(name: &str, fields: Vec<&str>) -> Decl {
    Decl { kind: DeclKind::Struct(StructDecl { name: ident(name), type_params: vec![], fields: fields.into_iter().map(|f| FieldDecl { name: ident(f), ty: dummy_type_expr(), default: None, span: span() }).collect() }), span: span() }
}

// -- Tests --------------------------------------------------------------

#[test] fn test_basic_arithmetic_int() {
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.eval_expr(&binary(BinOp::Add, int_expr(2), int_expr(3))).unwrap(), Value::Int(5)));
    assert!(matches!(interp.eval_expr(&binary(BinOp::Sub, int_expr(10), int_expr(4))).unwrap(), Value::Int(6)));
    assert!(matches!(interp.eval_expr(&binary(BinOp::Mul, int_expr(3), int_expr(7))).unwrap(), Value::Int(21)));
    assert!(matches!(interp.eval_expr(&binary(BinOp::Div, int_expr(15), int_expr(4))).unwrap(), Value::Int(3)));
    assert!(matches!(interp.eval_expr(&binary(BinOp::Mod, int_expr(17), int_expr(5))).unwrap(), Value::Int(2)));
}

#[test] fn test_basic_arithmetic_float() {
    let mut interp = Interpreter::new(0);
    match interp.eval_expr(&binary(BinOp::Add, float_expr(1.5), float_expr(2.5))).unwrap() { Value::Float(v) => assert!((v - 4.0).abs() < 1e-12), _ => panic!("expected Float") }
    match interp.eval_expr(&binary(BinOp::Mul, float_expr(3.0), float_expr(2.0))).unwrap() { Value::Float(v) => assert!((v - 6.0).abs() < 1e-12), _ => panic!("expected Float") }
}

#[test] fn test_comparison_operators() {
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.eval_expr(&binary(BinOp::Lt, int_expr(3), int_expr(5))).unwrap(), Value::Bool(true)));
    assert!(matches!(interp.eval_expr(&binary(BinOp::Ge, int_expr(3), int_expr(5))).unwrap(), Value::Bool(false)));
    assert!(matches!(interp.eval_expr(&binary(BinOp::Eq, int_expr(7), int_expr(7))).unwrap(), Value::Bool(true)));
    assert!(matches!(interp.eval_expr(&binary(BinOp::Ne, int_expr(7), int_expr(8))).unwrap(), Value::Bool(true)));
}

#[test] fn test_boolean_logic() {
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.eval_expr(&binary(BinOp::And, bool_expr(true), bool_expr(false))).unwrap(), Value::Bool(false)));
    assert!(matches!(interp.eval_expr(&binary(BinOp::Or, bool_expr(false), bool_expr(true))).unwrap(), Value::Bool(true)));
}

#[test] fn test_unary_operators() {
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.eval_expr(&unary(UnaryOp::Neg, int_expr(42))).unwrap(), Value::Int(-42)));
    assert!(matches!(interp.eval_expr(&unary(UnaryOp::Not, bool_expr(true))).unwrap(), Value::Bool(false)));
}

#[test] fn test_function_call() {
    let program = Program { declarations: vec![
        make_fn_decl("add", vec!["a", "b"], make_block(vec![], Some(binary(BinOp::Add, ident_expr("a"), ident_expr("b"))))),
        make_fn_decl("main", vec![], make_block(vec![], Some(call(ident_expr("add"), vec![int_expr(3), int_expr(4)])))),
    ]};
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.exec(&program).unwrap(), Value::Int(7)));
}

#[test] fn test_recursive_function() {
    let fact_body = make_block(vec![
        Stmt { kind: StmtKind::If(IfStmt { condition: binary(BinOp::Le, ident_expr("n"), int_expr(1)), then_block: make_block(vec![return_stmt(Some(int_expr(1)))], None), else_branch: None }), span: span() },
        return_stmt(Some(binary(BinOp::Mul, ident_expr("n"), call(ident_expr("factorial"), vec![binary(BinOp::Sub, ident_expr("n"), int_expr(1))])))),
    ], None);
    let program = Program { declarations: vec![
        make_fn_decl("factorial", vec!["n"], fact_body),
        make_fn_decl("main", vec![], make_block(vec![], Some(call(ident_expr("factorial"), vec![int_expr(5)])))),
    ]};
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.exec(&program).unwrap(), Value::Int(120)));
}

#[test] fn test_if_else() {
    let mut interp = Interpreter::new(0);
    let if_stmt = IfStmt { condition: bool_expr(true), then_block: make_block(vec![], Some(int_expr(42))), else_branch: Some(ElseBranch::Else(make_block(vec![], Some(int_expr(99))))) };
    assert!(matches!(interp.exec_if(&if_stmt).unwrap(), Value::Int(42)));
    let if_stmt_false = IfStmt { condition: bool_expr(false), then_block: make_block(vec![], Some(int_expr(42))), else_branch: Some(ElseBranch::Else(make_block(vec![], Some(int_expr(99))))) };
    assert!(matches!(interp.exec_if(&if_stmt_false).unwrap(), Value::Int(99)));
}

#[test] fn test_while_loop() {
    let program = Program { declarations: vec![make_fn_decl("main", vec![], make_block(vec![
        let_mut_stmt("i", int_expr(0)), let_mut_stmt("sum", int_expr(0)),
        Stmt { kind: StmtKind::While(WhileStmt { condition: binary(BinOp::Lt, ident_expr("i"), int_expr(5)), body: make_block(vec![
            expr_stmt(assign_expr(ident_expr("sum"), binary(BinOp::Add, ident_expr("sum"), ident_expr("i")))),
            expr_stmt(assign_expr(ident_expr("i"), binary(BinOp::Add, ident_expr("i"), int_expr(1)))),
        ], None) }), span: span() },
    ], Some(ident_expr("sum"))))]};
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.exec(&program).unwrap(), Value::Int(10)));
}

#[test] fn test_struct_creation_and_field_access() {
    let program = Program { declarations: vec![
        make_struct_decl("Point", vec!["x", "y"]),
        make_fn_decl("main", vec![], make_block(vec![let_stmt("p", struct_lit("Point", vec![("x", int_expr(10)), ("y", int_expr(20))]))], Some(binary(BinOp::Add, field_expr(ident_expr("p"), "x"), field_expr(ident_expr("p"), "y"))))),
    ]};
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.exec(&program).unwrap(), Value::Int(30)));
}

#[test] fn test_tensor_operations() {
    let mut interp = Interpreter::new(42);
    match &interp.eval_expr(&call(field_expr(ident_expr("Tensor"), "zeros"), vec![array_expr(vec![int_expr(2), int_expr(3)])])).unwrap() {
        Value::Tensor(t) => { assert_eq!(t.shape(), &[2, 3]); assert_eq!(t.len(), 6); assert!((t.sum() - 0.0).abs() < 1e-12); } _ => panic!("expected Tensor") }
    match &interp.eval_expr(&call(field_expr(ident_expr("Tensor"), "ones"), vec![array_expr(vec![int_expr(3)])])).unwrap() {
        Value::Tensor(t) => { assert_eq!(t.shape(), &[3]); assert!((t.sum() - 3.0).abs() < 1e-12); } _ => panic!("expected Tensor") }
    match &interp.eval_expr(&call(field_expr(ident_expr("Tensor"), "from_vec"), vec![array_expr(vec![float_expr(1.0), float_expr(2.0), float_expr(3.0), float_expr(4.0)]), array_expr(vec![int_expr(2), int_expr(2)])])).unwrap() {
        Value::Tensor(t) => { assert_eq!(t.shape(), &[2, 2]); assert!((t.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-12); assert!((t.get(&[1, 1]).unwrap() - 4.0).abs() < 1e-12); } _ => panic!("expected Tensor") }
}

#[test] fn test_tensor_arithmetic() {
    let mut interp = Interpreter::new(0);
    interp.define("a", Value::Tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap()));
    interp.define("b", Value::Tensor(Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap()));
    match &interp.eval_expr(&binary(BinOp::Add, ident_expr("a"), ident_expr("b"))).unwrap() { Value::Tensor(t) => assert_eq!(t.to_vec(), vec![5.0, 7.0, 9.0]), _ => panic!("expected Tensor") }
    match &interp.eval_expr(&binary(BinOp::Mul, ident_expr("a"), ident_expr("b"))).unwrap() { Value::Tensor(t) => assert_eq!(t.to_vec(), vec![4.0, 10.0, 18.0]), _ => panic!("expected Tensor") }
}

#[test] fn test_matmul() {
    let mut interp = Interpreter::new(0);
    interp.define("a", Value::Tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap()));
    interp.define("b", Value::Tensor(Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap()));
    match &interp.eval_expr(&call(ident_expr("matmul"), vec![ident_expr("a"), ident_expr("b")])).unwrap() {
        Value::Tensor(t) => { assert_eq!(t.shape(), &[2, 2]); assert!((t.get(&[0, 0]).unwrap() - 19.0).abs() < 1e-12); assert!((t.get(&[0, 1]).unwrap() - 22.0).abs() < 1e-12); assert!((t.get(&[1, 0]).unwrap() - 43.0).abs() < 1e-12); assert!((t.get(&[1, 1]).unwrap() - 50.0).abs() < 1e-12); } _ => panic!("expected Tensor") }
}

#[test] fn test_pipe_operator() {
    let program = Program { declarations: vec![
        make_fn_decl("double", vec!["x"], make_block(vec![], Some(binary(BinOp::Mul, ident_expr("x"), int_expr(2))))),
        make_fn_decl("add_one", vec!["x"], make_block(vec![], Some(binary(BinOp::Add, ident_expr("x"), int_expr(1))))),
        make_fn_decl("main", vec![], make_block(vec![], Some(pipe_expr(pipe_expr(int_expr(5), call(ident_expr("double"), vec![])), call(ident_expr("add_one"), vec![]))))),
    ]};
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.exec(&program).unwrap(), Value::Int(11)));
}

#[test] fn test_pipe_with_extra_args() {
    let program = Program { declarations: vec![
        make_fn_decl("add", vec!["a", "b"], make_block(vec![], Some(binary(BinOp::Add, ident_expr("a"), ident_expr("b"))))),
        make_fn_decl("main", vec![], make_block(vec![], Some(pipe_expr(int_expr(10), call(ident_expr("add"), vec![int_expr(5)]))))),
    ]};
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.exec(&program).unwrap(), Value::Int(15)));
}

#[test] fn test_print_builtin() {
    let mut interp = Interpreter::new(0);
    interp.eval_expr(&call(ident_expr("print"), vec![string_expr("hello"), int_expr(42)])).unwrap();
    assert_eq!(interp.output.len(), 1);
    assert_eq!(interp.output[0], "hello 42");
}

#[test] fn test_array_literal_and_indexing() {
    let mut interp = Interpreter::new(0);
    interp.define("arr", Value::Array(std::rc::Rc::new(vec![Value::Int(10), Value::Int(20), Value::Int(30)])));
    assert!(matches!(interp.eval_expr(&index_expr(ident_expr("arr"), int_expr(1))).unwrap(), Value::Int(20)));
}

#[test] fn test_tensor_multi_index() {
    let mut interp = Interpreter::new(0);
    interp.define("t", Value::Tensor(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap()));
    match interp.eval_expr(&multi_index_expr(ident_expr("t"), vec![int_expr(1), int_expr(2)])).unwrap() { Value::Float(v) => assert!((v - 6.0).abs() < 1e-12), _ => panic!("expected Float") }
}

#[test] fn test_variable_assignment() {
    let program = Program { declarations: vec![make_fn_decl("main", vec![], make_block(vec![let_mut_stmt("x", int_expr(10)), expr_stmt(assign_expr(ident_expr("x"), int_expr(20)))], Some(ident_expr("x"))))] };
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.exec(&program).unwrap(), Value::Int(20)));
}

#[test] fn test_nested_scopes() {
    let program = Program { declarations: vec![make_fn_decl("main", vec![], make_block(vec![
        let_stmt("x", int_expr(1)),
        let_stmt("y", Expr { kind: ExprKind::Block(make_block(vec![let_stmt("x", int_expr(99))], Some(ident_expr("x")))), span: span() }),
    ], Some(binary(BinOp::Add, ident_expr("x"), ident_expr("y")))))] };
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.exec(&program).unwrap(), Value::Int(100)));
}

#[test] fn test_early_return() {
    let func = make_fn_decl("early", vec!["x"], make_block(vec![
        Stmt { kind: StmtKind::If(IfStmt { condition: binary(BinOp::Gt, ident_expr("x"), int_expr(0)), then_block: make_block(vec![return_stmt(Some(ident_expr("x")))], None), else_branch: None }), span: span() },
        return_stmt(Some(int_expr(0))),
    ], None));
    let program = Program { declarations: vec![func, make_fn_decl("main", vec![], make_block(vec![], Some(binary(BinOp::Add, call(ident_expr("early"), vec![int_expr(5)]), call(ident_expr("early"), vec![int_expr(-3)])))))] };
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.exec(&program).unwrap(), Value::Int(5)));
}

#[test] fn test_string_concatenation() {
    let mut interp = Interpreter::new(0);
    match interp.eval_expr(&binary(BinOp::Add, string_expr("hello "), string_expr("world"))).unwrap() { Value::String(s) => assert_eq!(s.as_str(), "hello world"), _ => panic!("expected String") }
}

#[test] fn test_division_by_zero() {
    let mut interp = Interpreter::new(0);
    assert!(interp.eval_expr(&binary(BinOp::Div, int_expr(10), int_expr(0))).is_err());
}

#[test] fn test_undefined_variable() {
    let mut interp = Interpreter::new(0);
    assert!(interp.eval_expr(&ident_expr("nonexistent")).is_err());
}

#[test] fn test_tensor_randn_deterministic() {
    let mut interp1 = Interpreter::new(42);
    let mut interp2 = Interpreter::new(42);
    let create_randn = call(field_expr(ident_expr("Tensor"), "randn"), vec![array_expr(vec![int_expr(3), int_expr(4)])]);
    match (&interp1.eval_expr(&create_randn).unwrap(), &interp2.eval_expr(&create_randn).unwrap()) { (Value::Tensor(a), Value::Tensor(b)) => assert_eq!(a.to_vec(), b.to_vec()), _ => panic!("expected Tensors") }
}

#[test] fn test_if_else_chain() {
    let mut interp = Interpreter::new(0);
    let if_stmt = IfStmt { condition: bool_expr(false), then_block: make_block(vec![], Some(int_expr(1))), else_branch: Some(ElseBranch::ElseIf(Box::new(IfStmt { condition: bool_expr(false), then_block: make_block(vec![], Some(int_expr(2))), else_branch: Some(ElseBranch::Else(make_block(vec![], Some(int_expr(3))))) }))) };
    assert!(matches!(interp.exec_if(&if_stmt).unwrap(), Value::Int(3)));
}

#[test] fn test_struct_field_assignment() {
    let program = Program { declarations: vec![
        make_struct_decl("Point", vec!["x", "y"]),
        make_fn_decl("main", vec![], make_block(vec![
            let_mut_stmt("p", struct_lit("Point", vec![("x", int_expr(10)), ("y", int_expr(20))])),
            expr_stmt(assign_expr(field_expr(ident_expr("p"), "x"), int_expr(42))),
        ], Some(field_expr(ident_expr("p"), "x")))),
    ]};
    let mut interp = Interpreter::new(0);
    assert!(matches!(interp.exec(&program).unwrap(), Value::Int(42)));
}

#[test] fn test_mixed_int_float_arithmetic() {
    let mut interp = Interpreter::new(0);
    match interp.eval_expr(&binary(BinOp::Add, int_expr(2), float_expr(3.5))).unwrap() { Value::Float(v) => assert!((v - 5.5).abs() < 1e-12), _ => panic!("expected Float") }
}
