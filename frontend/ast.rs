// frontend/src/ast.rs

//! Taria AST: Immutable, arena-allocated, source-mapped

use std::rc::Rc;
use std::ops::Range;

pub type Span = Range<usize>;

#[derive(Debug, Clone)]
pub enum ExprKind {
    Identifier(String),
    Call { func: Box<Expr>, args: Vec<Expr> },
    Literal(String),
    Tensor(String),
    // ... extensible
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Decorator {
    pub name: String,
    pub args: Vec<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub ty: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub name: String,
    pub decorators: Vec<Decorator>,
    pub params: Vec<Parameter>,
    pub return_type: Option<String>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    Return(Expr),
    Let { name: String, expr: Expr },
    Expr(Expr),
    // ... extensible
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ModuleAST {
    pub functions: Vec<FunctionDecl>,
    pub span: Span,
}

// Visitor pattern for AST traversal
pub trait AstVisitor {
    fn visit_module(&mut self, module: &ModuleAST);
    fn visit_function(&mut self, func: &FunctionDecl);
    fn visit_stmt(&mut self, stmt: &Stmt);
    fn visit_expr(&mut self, expr: &Expr);
    // ... extensible
}

// Arena allocation: Use bumpalo or typed-arena for AST node storage.
// Source mapping: All nodes carry spans for diagnostics.
// Diagnostics: AST nodes can emit errors via hooks.
