use std::fmt;

/// Source span: byte offset range in source code.
/// Duplicated from cjc-diag to avoid circular dependency.
/// AST is a leaf crate with no dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }
}

// ── AST Node Types ──────────────────────────────────────────────

/// Top-level program: a sequence of declarations.
#[derive(Debug, Clone)]
pub struct Program {
    pub declarations: Vec<Decl>,
}

/// A declaration (top-level item).
#[derive(Debug, Clone)]
pub struct Decl {
    pub kind: DeclKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum DeclKind {
    Struct(StructDecl),
    Class(ClassDecl),
    Fn(FnDecl),
    Trait(TraitDecl),
    Impl(ImplDecl),
    Enum(EnumDecl),
    Let(LetStmt),
    Import(ImportDecl),
    /// Compile-time constant: `const NAME: Type = expr;`
    Const(ConstDecl),
    /// Top-level statement (expression statement, if, while, etc.)
    Stmt(Stmt),
}

/// A compile-time constant declaration: `const NAME: Type = expr;`
#[derive(Debug, Clone)]
pub struct ConstDecl {
    pub name: Ident,
    pub ty: TypeExpr,
    pub value: Box<Expr>,
    pub span: Span,
}

// ── Struct & Class ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StructDecl {
    pub name: Ident,
    pub type_params: Vec<TypeParam>,
    pub fields: Vec<FieldDecl>,
}

#[derive(Debug, Clone)]
pub struct ClassDecl {
    pub name: Ident,
    pub type_params: Vec<TypeParam>,
    pub fields: Vec<FieldDecl>,
}

// ── Enums ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: Ident,
    pub type_params: Vec<TypeParam>,
    pub variants: Vec<VariantDecl>,
}

#[derive(Debug, Clone)]
pub struct VariantDecl {
    pub name: Ident,
    /// Payload types for tuple-like variants. Empty for unit variants.
    pub fields: Vec<TypeExpr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FieldDecl {
    pub name: Ident,
    pub ty: TypeExpr,
    pub default: Option<Expr>,
    pub span: Span,
}

// ── Functions ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: Ident,
    pub type_params: Vec<TypeParam>,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub body: Block,
    pub is_nogc: bool,
}

#[derive(Debug, Clone)]
pub struct FnSig {
    pub name: Ident,
    pub type_params: Vec<TypeParam>,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: Ident,
    pub ty: TypeExpr,
    pub span: Span,
}

// ── Traits & Impls ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TraitDecl {
    pub name: Ident,
    pub type_params: Vec<TypeParam>,
    pub super_traits: Vec<TypeExpr>,
    pub methods: Vec<FnSig>,
}

#[derive(Debug, Clone)]
pub struct ImplDecl {
    pub type_params: Vec<TypeParam>,
    pub target: TypeExpr,
    pub trait_ref: Option<TypeExpr>,
    pub methods: Vec<FnDecl>,
    pub span: Span,
}

// ── Import ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImportDecl {
    pub path: Vec<Ident>,
    pub alias: Option<Ident>,
}

// ── Type Expressions ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TypeExpr {
    pub kind: TypeExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeExprKind {
    /// Named type: `f64`, `Tensor<f32>`, etc.
    Named {
        name: Ident,
        args: Vec<TypeArg>,
    },
    /// Array type: `[T; N]`
    Array {
        elem: Box<TypeExpr>,
        size: Box<Expr>,
    },
    /// Tuple type: `(T, U)`
    Tuple(Vec<TypeExpr>),
    /// Function type: `fn(T, U) -> V`
    Fn {
        params: Vec<TypeExpr>,
        ret: Box<TypeExpr>,
    },
    /// Shape literal in type position: `[M, N]`
    ShapeLit(Vec<ShapeDim>),
}

/// A type argument can be a type or an expression (for shape params).
#[derive(Debug, Clone)]
pub enum TypeArg {
    Type(TypeExpr),
    Expr(Expr),
    Shape(Vec<ShapeDim>),
}

/// A shape dimension: either a symbolic name or a literal integer.
#[derive(Debug, Clone)]
pub enum ShapeDim {
    Name(Ident),
    Lit(i64),
}

#[derive(Debug, Clone)]
pub struct TypeParam {
    pub name: Ident,
    pub bounds: Vec<TypeExpr>,
    pub span: Span,
}

// ── Statements ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub expr: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    Let(LetStmt),
    Expr(Expr),
    Return(Option<Expr>),
    Break,
    Continue,
    If(IfStmt),
    While(WhileStmt),
    For(ForStmt),
    NoGcBlock(Block),
}

#[derive(Debug, Clone)]
pub struct LetStmt {
    pub name: Ident,
    pub mutable: bool,
    pub ty: Option<TypeExpr>,
    pub init: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct IfStmt {
    pub condition: Expr,
    pub then_block: Block,
    pub else_branch: Option<ElseBranch>,
}

#[derive(Debug, Clone)]
pub enum ElseBranch {
    ElseIf(Box<IfStmt>),
    Else(Block),
}

#[derive(Debug, Clone)]
pub struct WhileStmt {
    pub condition: Expr,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct ForStmt {
    pub ident: Ident,
    pub iter: ForIter,
    pub body: Block,
}

/// The iteration source for a `for` loop.
#[derive(Debug, Clone)]
pub enum ForIter {
    /// Range form: `start..end` (exclusive end)
    Range { start: Box<Expr>, end: Box<Expr> },
    /// Expression form: `for x in expr` (array/tensor iteration)
    Expr(Box<Expr>),
}

// ── Expressions ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Integer literal: `42`
    IntLit(i64),
    /// Float literal: `3.14`
    FloatLit(f64),
    /// String literal: `"hello"`
    StringLit(String),
    /// Byte string literal: `b"hello"` — produces ByteSlice
    ByteStringLit(Vec<u8>),
    /// Byte char literal: `b'A'` — produces u8
    ByteCharLit(u8),
    /// Raw string literal: `r"hello\n"` — produces String (no escape processing)
    RawStringLit(String),
    /// Raw byte string literal: `br"hello\n"` — produces ByteSlice (no escape processing)
    RawByteStringLit(Vec<u8>),
    /// Format string literal: `f"hello {name}!"` — desugared into concat at parse time.
    /// Stored as alternating literal segments and expression strings to format.
    /// Each segment: `(literal_text, Option<expr>)` — literal then optional interp.
    FStringLit(Vec<(String, Option<Box<Expr>>)>),
    /// Regex literal: `/pattern/flags` — produces Regex
    RegexLit { pattern: String, flags: String },
    /// Tensor literal: `[| 1.0, 2.0; 3.0, 4.0 |]` — produces Tensor
    /// `rows` contains each row as a Vec of expressions.
    /// A 1-D tensor has a single row. `;` separates rows for 2-D.
    TensorLit { rows: Vec<Vec<Expr>> },
    /// Bool literal: `true`, `false`
    BoolLit(bool),
    /// Identifier: `x`, `foo`
    Ident(Ident),
    /// Binary operation: `a + b`
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Unary operation: `-x`, `!b`
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    /// Function/method call: `foo(a, b)`
    Call {
        callee: Box<Expr>,
        args: Vec<CallArg>,
    },
    /// Field access / method access: `x.foo`
    Field {
        object: Box<Expr>,
        name: Ident,
    },
    /// Index: `a[i]`
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
    },
    /// Multi-index: `a[i, j]`
    MultiIndex {
        object: Box<Expr>,
        indices: Vec<Expr>,
    },
    /// Assignment: `x = expr`
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },
    /// Compound assignment: `x += e`, `x -= e`, etc.
    /// Desugars to `x = x op e` during evaluation.
    CompoundAssign {
        op: BinOp,
        target: Box<Expr>,
        value: Box<Expr>,
    },
    /// If expression: `if cond { a } else { b }` used as a value
    IfExpr {
        condition: Box<Expr>,
        then_block: Block,
        else_branch: Option<ElseBranch>,
    },
    /// Pipe: `a |> f(b)`
    Pipe {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Block expression: `{ stmts; expr }`
    Block(Block),
    /// Struct literal: `Foo { x: 1, y: 2 }`
    StructLit {
        name: Ident,
        fields: Vec<FieldInit>,
    },
    /// Array literal: `[1, 2, 3]`
    ArrayLit(Vec<Expr>),
    /// Column reference in data DSL: `col("name")`
    Col(String),
    /// Lambda expression: `|x, y| x + y`
    Lambda {
        params: Vec<Param>,
        body: Box<Expr>,
    },
    /// Match expression: `match expr { pat => expr, ... }`
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    /// Tuple literal: `(a, b, c)`
    TupleLit(Vec<Expr>),
    /// Try operator: `expr?` — desugars to match on Result
    Try(Box<Expr>),
    /// Enum variant constructor: `Some(42)` or `None`
    VariantLit {
        enum_name: Option<Ident>,
        variant: Ident,
        fields: Vec<Expr>,
    },
}

// ── Match Arms & Patterns ───────────────────────────────────────

/// A single arm of a match expression: `pattern => body`
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

/// A pattern for use in match arms.
#[derive(Debug, Clone)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum PatternKind {
    /// Wildcard pattern: `_`
    Wildcard,
    /// Binding pattern: `x` (binds the matched value to `x`)
    Binding(Ident),
    /// Literal pattern: `42`, `3.14`, `true`, `"hello"`
    LitInt(i64),
    LitFloat(f64),
    LitBool(bool),
    LitString(String),
    /// Tuple destructuring: `(a, b, c)`
    Tuple(Vec<Pattern>),
    /// Struct destructuring: `Point { x, y }` or `Point { x: px, y: py }`
    Struct {
        name: Ident,
        fields: Vec<PatternField>,
    },
    /// Enum variant pattern: `Some(x)`, `None`, `Ok(v)`, `Err(e)`
    Variant {
        /// Optional enum name qualification (not needed for prelude enums)
        enum_name: Option<Ident>,
        variant: Ident,
        fields: Vec<Pattern>,
    },
}

/// A field pattern inside a struct pattern: `x` or `x: pat`
#[derive(Debug, Clone)]
pub struct PatternField {
    pub name: Ident,
    pub pattern: Option<Pattern>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct CallArg {
    pub name: Option<Ident>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FieldInit {
    pub name: Ident,
    pub value: Expr,
    pub span: Span,
}

// ── Operators ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,       // **
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Match,     // ~=  (regex binding)
    NotMatch,  // !~  (negative regex binding)
    // Bitwise operators
    BitAnd,    // &
    BitOr,     // |
    BitXor,    // ^
    Shl,       // <<
    Shr,       // >>
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::Pow => write!(f, "**"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Ne => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Le => write!(f, "<="),
            BinOp::Ge => write!(f, ">="),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::Match => write!(f, "~="),
            BinOp::NotMatch => write!(f, "!~"),
            BinOp::BitAnd => write!(f, "&"),
            BinOp::BitOr => write!(f, "|"),
            BinOp::BitXor => write!(f, "^"),
            BinOp::Shl => write!(f, "<<"),
            BinOp::Shr => write!(f, ">>"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,  // ~ (bitwise NOT)
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
            UnaryOp::BitNot => write!(f, "~"),
        }
    }
}

// ── Identifier ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

impl Ident {
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }

    pub fn dummy(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            span: Span::dummy(),
        }
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ── Pretty Printer ──────────────────────────────────────────────

pub struct PrettyPrinter {
    indent: usize,
    output: String,
}

impl PrettyPrinter {
    pub fn new() -> Self {
        Self {
            indent: 0,
            output: String::new(),
        }
    }

    pub fn print_program(mut self, program: &Program) -> String {
        for decl in &program.declarations {
            self.print_decl(decl);
            self.output.push('\n');
        }
        self.output
    }

    fn indent_str(&self) -> String {
        "    ".repeat(self.indent)
    }

    fn print_decl(&mut self, decl: &Decl) {
        match &decl.kind {
            DeclKind::Struct(s) => self.print_struct(s),
            DeclKind::Class(c) => self.print_class(c),
            DeclKind::Fn(f) => self.print_fn(f),
            DeclKind::Trait(t) => self.print_trait(t),
            DeclKind::Impl(i) => self.print_impl(i),
            DeclKind::Enum(e) => self.print_enum(e),
            DeclKind::Let(l) => {
                self.output.push_str(&self.indent_str());
                self.print_let(l);
            }
            DeclKind::Import(i) => self.print_import(i),
            DeclKind::Const(c) => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("const ");
                self.output.push_str(&c.name.name);
                self.output.push_str(": ");
                self.print_type_expr(&c.ty);
                self.output.push_str(" = ");
                self.print_expr(&c.value);
                self.output.push(';');
            }
            DeclKind::Stmt(s) => {
                self.output.push_str(&self.indent_str());
                self.print_stmt(s);
            }
        }
    }

    fn print_struct(&mut self, s: &StructDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str(&format!("struct {}", s.name));
        self.print_type_params(&s.type_params);
        self.output.push_str(" {\n");
        self.indent += 1;
        for field in &s.fields {
            self.output
                .push_str(&format!("{}{}: ", self.indent_str(), field.name));
            self.print_type_expr(&field.ty);
            self.output.push('\n');
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_class(&mut self, c: &ClassDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str(&format!("class {}", c.name));
        self.print_type_params(&c.type_params);
        self.output.push_str(" {\n");
        self.indent += 1;
        for field in &c.fields {
            self.output
                .push_str(&format!("{}{}: ", self.indent_str(), field.name));
            self.print_type_expr(&field.ty);
            self.output.push('\n');
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_fn(&mut self, f: &FnDecl) {
        self.output.push_str(&self.indent_str());
        if f.is_nogc {
            self.output.push_str("nogc ");
        }
        self.output.push_str(&format!("fn {}", f.name));
        self.print_type_params(&f.type_params);
        self.output.push('(');
        for (i, param) in f.params.iter().enumerate() {
            if i > 0 {
                self.output.push_str(", ");
            }
            self.output.push_str(&format!("{}: ", param.name));
            self.print_type_expr(&param.ty);
        }
        self.output.push(')');
        if let Some(ref ret) = f.return_type {
            self.output.push_str(" -> ");
            self.print_type_expr(ret);
        }
        self.output.push(' ');
        self.print_block(&f.body);
    }

    fn print_trait(&mut self, t: &TraitDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str(&format!("trait {}", t.name));
        self.print_type_params(&t.type_params);
        if !t.super_traits.is_empty() {
            self.output.push_str(": ");
            for (i, st) in t.super_traits.iter().enumerate() {
                if i > 0 {
                    self.output.push_str(" + ");
                }
                self.print_type_expr(st);
            }
        }
        self.output.push_str(" {\n");
        self.indent += 1;
        for method in &t.methods {
            self.output.push_str(&self.indent_str());
            self.output.push_str(&format!("fn {}", method.name));
            self.print_type_params(&method.type_params);
            self.output.push('(');
            for (i, p) in method.params.iter().enumerate() {
                if i > 0 {
                    self.output.push_str(", ");
                }
                self.output.push_str(&format!("{}: ", p.name));
                self.print_type_expr(&p.ty);
            }
            self.output.push(')');
            if let Some(ref ret) = method.return_type {
                self.output.push_str(" -> ");
                self.print_type_expr(ret);
            }
            self.output.push_str(";\n");
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_impl(&mut self, i: &ImplDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str("impl");
        self.print_type_params(&i.type_params);
        self.output.push(' ');
        self.print_type_expr(&i.target);
        if let Some(ref tr) = i.trait_ref {
            self.output.push_str(" : ");
            self.print_type_expr(tr);
        }
        self.output.push_str(" {\n");
        self.indent += 1;
        for method in &i.methods {
            self.print_fn(method);
            self.output.push('\n');
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_import(&mut self, i: &ImportDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str("import ");
        let path: Vec<&str> = i.path.iter().map(|id| id.name.as_str()).collect();
        self.output.push_str(&path.join("."));
        if let Some(ref alias) = i.alias {
            self.output.push_str(&format!(" as {}", alias));
        }
    }

    fn print_enum(&mut self, e: &EnumDecl) {
        self.output.push_str(&self.indent_str());
        self.output.push_str(&format!("enum {}", e.name));
        self.print_type_params(&e.type_params);
        self.output.push_str(" {\n");
        self.indent += 1;
        for variant in &e.variants {
            self.output.push_str(&self.indent_str());
            self.output.push_str(&variant.name.name);
            if !variant.fields.is_empty() {
                self.output.push('(');
                for (i, f) in variant.fields.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_type_expr(f);
                }
                self.output.push(')');
            }
            self.output.push_str(",\n");
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_type_params(&mut self, params: &[TypeParam]) {
        if params.is_empty() {
            return;
        }
        self.output.push('<');
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                self.output.push_str(", ");
            }
            self.output.push_str(&param.name.name);
            if !param.bounds.is_empty() {
                self.output.push_str(": ");
                for (j, bound) in param.bounds.iter().enumerate() {
                    if j > 0 {
                        self.output.push_str(" + ");
                    }
                    self.print_type_expr(bound);
                }
            }
        }
        self.output.push('>');
    }

    fn print_type_expr(&mut self, ty: &TypeExpr) {
        match &ty.kind {
            TypeExprKind::Named { name, args } => {
                self.output.push_str(&name.name);
                if !args.is_empty() {
                    self.output.push('<');
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            self.output.push_str(", ");
                        }
                        match arg {
                            TypeArg::Type(t) => self.print_type_expr(t),
                            TypeArg::Expr(e) => self.print_expr(e),
                            TypeArg::Shape(dims) => self.print_shape(dims),
                        }
                    }
                    self.output.push('>');
                }
            }
            TypeExprKind::Array { elem, size } => {
                self.output.push('[');
                self.print_type_expr(elem);
                self.output.push_str("; ");
                self.print_expr(size);
                self.output.push(']');
            }
            TypeExprKind::Tuple(elems) => {
                self.output.push('(');
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_type_expr(elem);
                }
                self.output.push(')');
            }
            TypeExprKind::Fn { params, ret } => {
                self.output.push_str("fn(");
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_type_expr(p);
                }
                self.output.push_str(") -> ");
                self.print_type_expr(ret);
            }
            TypeExprKind::ShapeLit(dims) => {
                self.print_shape(dims);
            }
        }
    }

    fn print_shape(&mut self, dims: &[ShapeDim]) {
        self.output.push('[');
        for (i, dim) in dims.iter().enumerate() {
            if i > 0 {
                self.output.push_str(", ");
            }
            match dim {
                ShapeDim::Name(n) => self.output.push_str(&n.name),
                ShapeDim::Lit(v) => self.output.push_str(&v.to_string()),
            }
        }
        self.output.push(']');
    }

    fn print_block(&mut self, block: &Block) {
        self.output.push_str("{\n");
        self.indent += 1;
        for stmt in &block.stmts {
            self.print_stmt(stmt);
        }
        if let Some(ref expr) = block.expr {
            self.output.push_str(&self.indent_str());
            self.print_expr(expr);
            self.output.push('\n');
        }
        self.indent -= 1;
        self.output.push_str(&format!("{}}}", self.indent_str()));
    }

    fn print_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let(l) => {
                self.output.push_str(&self.indent_str());
                self.print_let(l);
            }
            StmtKind::Expr(e) => {
                self.output.push_str(&self.indent_str());
                self.print_expr(e);
                self.output.push_str(";\n");
            }
            StmtKind::Return(e) => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("return");
                if let Some(expr) = e {
                    self.output.push(' ');
                    self.print_expr(expr);
                }
                self.output.push_str(";\n");
            }
            StmtKind::Break => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("break;\n");
            }
            StmtKind::Continue => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("continue;\n");
            }
            StmtKind::If(if_stmt) => {
                self.output.push_str(&self.indent_str());
                self.print_if(if_stmt);
                self.output.push('\n');
            }
            StmtKind::While(w) => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("while ");
                self.print_expr(&w.condition);
                self.output.push(' ');
                self.print_block(&w.body);
                self.output.push('\n');
            }
            StmtKind::For(f) => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("for ");
                self.output.push_str(&f.ident.name);
                self.output.push_str(" in ");
                match &f.iter {
                    ForIter::Range { start, end } => {
                        self.print_expr(start);
                        self.output.push_str("..");
                        self.print_expr(end);
                    }
                    ForIter::Expr(expr) => {
                        self.print_expr(expr);
                    }
                }
                self.output.push(' ');
                self.print_block(&f.body);
                self.output.push('\n');
            }
            StmtKind::NoGcBlock(block) => {
                self.output.push_str(&self.indent_str());
                self.output.push_str("nogc ");
                self.print_block(block);
                self.output.push('\n');
            }
        }
    }

    fn print_let(&mut self, l: &LetStmt) {
        self.output.push_str("let ");
        if l.mutable {
            self.output.push_str("mut ");
        }
        self.output.push_str(&l.name.name);
        if let Some(ref ty) = l.ty {
            self.output.push_str(": ");
            self.print_type_expr(ty);
        }
        self.output.push_str(" = ");
        self.print_expr(&l.init);
        self.output.push_str(";\n");
    }

    fn print_if(&mut self, if_stmt: &IfStmt) {
        self.output.push_str("if ");
        self.print_expr(&if_stmt.condition);
        self.output.push(' ');
        self.print_block(&if_stmt.then_block);
        if let Some(ref else_branch) = if_stmt.else_branch {
            self.output.push_str(" else ");
            match else_branch {
                ElseBranch::ElseIf(elif) => self.print_if(elif),
                ElseBranch::Else(block) => self.print_block(block),
            }
        }
    }

    fn print_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::IntLit(v) => self.output.push_str(&v.to_string()),
            ExprKind::FloatLit(v) => {
                let s = v.to_string();
                self.output.push_str(&s);
                if !s.contains('.') {
                    self.output.push_str(".0");
                }
            }
            ExprKind::StringLit(s) => {
                self.output.push('"');
                self.output.push_str(s);
                self.output.push('"');
            }
            ExprKind::ByteStringLit(bytes) => {
                self.output.push_str("b\"");
                for &b in bytes {
                    if b.is_ascii_graphic() || b == b' ' {
                        self.output.push(b as char);
                    } else {
                        self.output.push_str(&format!("\\x{:02x}", b));
                    }
                }
                self.output.push('"');
            }
            ExprKind::ByteCharLit(b) => {
                self.output.push_str(&format!("b'\\x{:02x}'", b));
            }
            ExprKind::RawStringLit(s) => {
                self.output.push_str("r\"");
                self.output.push_str(s);
                self.output.push('"');
            }
            ExprKind::RawByteStringLit(bytes) => {
                self.output.push_str("br\"");
                for &b in bytes {
                    self.output.push(b as char);
                }
                self.output.push('"');
            }
            ExprKind::FStringLit(segments) => {
                // Pretty-print as f"..."
                self.output.push_str("f\"");
                for (lit, interp) in segments {
                    self.output.push_str(&lit.replace('"', "\\\""));
                    if let Some(expr) = interp {
                        self.output.push('{');
                        self.print_expr(expr);
                        self.output.push('}');
                    }
                }
                self.output.push('"');
            }
            ExprKind::RegexLit { pattern, flags } => {
                self.output.push('/');
                self.output.push_str(pattern);
                self.output.push('/');
                self.output.push_str(flags);
            }
            ExprKind::TensorLit { rows } => {
                self.output.push_str("[| ");
                for (ri, row) in rows.iter().enumerate() {
                    if ri > 0 {
                        self.output.push_str("; ");
                    }
                    for (ci, expr) in row.iter().enumerate() {
                        if ci > 0 {
                            self.output.push_str(", ");
                        }
                        self.print_expr(expr);
                    }
                }
                self.output.push_str(" |]");
            }
            ExprKind::BoolLit(b) => self.output.push_str(if *b { "true" } else { "false" }),
            ExprKind::Ident(id) => self.output.push_str(&id.name),
            ExprKind::Binary { op, left, right } => {
                self.output.push('(');
                self.print_expr(left);
                self.output.push_str(&format!(" {} ", op));
                self.print_expr(right);
                self.output.push(')');
            }
            ExprKind::Unary { op, operand } => {
                self.output.push_str(&format!("{}", op));
                self.print_expr(operand);
            }
            ExprKind::Call { callee, args } => {
                self.print_expr(callee);
                self.output.push('(');
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    if let Some(ref name) = arg.name {
                        self.output.push_str(&format!("{}: ", name));
                    }
                    self.print_expr(&arg.value);
                }
                self.output.push(')');
            }
            ExprKind::Field { object, name } => {
                self.print_expr(object);
                self.output.push('.');
                self.output.push_str(&name.name);
            }
            ExprKind::Index { object, index } => {
                self.print_expr(object);
                self.output.push('[');
                self.print_expr(index);
                self.output.push(']');
            }
            ExprKind::MultiIndex { object, indices } => {
                self.print_expr(object);
                self.output.push('[');
                for (i, idx) in indices.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_expr(idx);
                }
                self.output.push(']');
            }
            ExprKind::Assign { target, value } => {
                self.print_expr(target);
                self.output.push_str(" = ");
                self.print_expr(value);
            }
            ExprKind::CompoundAssign { op, target, value } => {
                self.print_expr(target);
                self.output.push_str(&format!(" {}= ", op));
                self.print_expr(value);
            }
            ExprKind::IfExpr { condition, then_block, else_branch } => {
                self.output.push_str("if ");
                self.print_expr(condition);
                self.output.push_str(" ");
                self.print_block(then_block);
                if let Some(eb) = else_branch {
                    self.output.push_str(" else ");
                    match eb {
                        ElseBranch::ElseIf(elif) => {
                            // Print as if statement
                            self.output.push_str("if ...");
                        }
                        ElseBranch::Else(block) => {
                            self.print_block(block);
                        }
                    }
                }
            }
            ExprKind::Pipe { left, right } => {
                self.print_expr(left);
                self.output.push_str(" |> ");
                self.print_expr(right);
            }
            ExprKind::Block(block) => {
                self.print_block(block);
            }
            ExprKind::StructLit { name, fields } => {
                self.output.push_str(&format!("{} {{ ", name));
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.output.push_str(&format!("{}: ", field.name));
                    self.print_expr(&field.value);
                }
                self.output.push_str(" }");
            }
            ExprKind::ArrayLit(elems) => {
                self.output.push('[');
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_expr(elem);
                }
                self.output.push(']');
            }
            ExprKind::Col(name) => {
                self.output.push_str(&format!("col(\"{}\")", name));
            }
            ExprKind::Lambda { params, body } => {
                self.output.push('|');
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.output.push_str(&format!("{}: ", p.name));
                    self.print_type_expr(&p.ty);
                }
                self.output.push_str("| ");
                self.print_expr(body);
            }
            ExprKind::Match { scrutinee, arms } => {
                self.output.push_str("match ");
                self.print_expr(scrutinee);
                self.output.push_str(" {\n");
                self.indent += 1;
                for arm in arms {
                    self.output.push_str(&self.indent_str());
                    self.print_pattern(&arm.pattern);
                    self.output.push_str(" => ");
                    self.print_expr(&arm.body);
                    self.output.push_str(",\n");
                }
                self.indent -= 1;
                self.output.push_str(&format!("{}}}", self.indent_str()));
            }
            ExprKind::TupleLit(elems) => {
                self.output.push('(');
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_expr(elem);
                }
                self.output.push(')');
            }
            ExprKind::Try(inner) => {
                self.print_expr(inner);
                self.output.push('?');
            }
            ExprKind::VariantLit {
                enum_name,
                variant,
                fields,
            } => {
                if let Some(ref en) = enum_name {
                    self.output.push_str(&en.name);
                    self.output.push_str("::");
                }
                self.output.push_str(&variant.name);
                if !fields.is_empty() {
                    self.output.push('(');
                    for (i, f) in fields.iter().enumerate() {
                        if i > 0 {
                            self.output.push_str(", ");
                        }
                        self.print_expr(f);
                    }
                    self.output.push(')');
                }
            }
        }
    }

    fn print_pattern(&mut self, pattern: &Pattern) {
        match &pattern.kind {
            PatternKind::Wildcard => self.output.push('_'),
            PatternKind::Binding(id) => self.output.push_str(&id.name),
            PatternKind::LitInt(v) => self.output.push_str(&v.to_string()),
            PatternKind::LitFloat(v) => self.output.push_str(&v.to_string()),
            PatternKind::LitBool(b) => {
                self.output.push_str(if *b { "true" } else { "false" })
            }
            PatternKind::LitString(s) => {
                self.output.push('"');
                self.output.push_str(s);
                self.output.push('"');
            }
            PatternKind::Tuple(pats) => {
                self.output.push('(');
                for (i, p) in pats.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.print_pattern(p);
                }
                self.output.push(')');
            }
            PatternKind::Struct { name, fields } => {
                self.output.push_str(&name.name);
                self.output.push_str(" { ");
                for (i, f) in fields.iter().enumerate() {
                    if i > 0 {
                        self.output.push_str(", ");
                    }
                    self.output.push_str(&f.name.name);
                    if let Some(ref pat) = f.pattern {
                        self.output.push_str(": ");
                        self.print_pattern(pat);
                    }
                }
                self.output.push_str(" }");
            }
            PatternKind::Variant {
                enum_name,
                variant,
                fields,
            } => {
                if let Some(ref en) = enum_name {
                    self.output.push_str(&en.name);
                    self.output.push_str("::");
                }
                self.output.push_str(&variant.name);
                if !fields.is_empty() {
                    self.output.push('(');
                    for (i, p) in fields.iter().enumerate() {
                        if i > 0 {
                            self.output.push_str(", ");
                        }
                        self.print_pattern(p);
                    }
                    self.output.push(')');
                }
            }
        }
    }
}

impl Default for PrettyPrinter {
    fn default() -> Self {
        Self::new()
    }
}
