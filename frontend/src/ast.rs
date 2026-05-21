//! Taria AST: Immutable, arena-allocated (conceptually), source-mapped

use crate::source_map::Span;

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
