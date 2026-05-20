//! Taria AST: Immutable, arena-allocated (conceptually), source-mapped

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

/// Visitor pattern for AST traversal
pub trait AstVisitor {
    fn visit_module(&mut self, module: &ModuleAST) {
        for func in &module.functions {
            self.visit_function(func);
        }
    }

    fn visit_function(&mut self, func: &FunctionDecl) {
        for stmt in &func.body {
            self.visit_stmt(stmt);
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Return(expr) => self.visit_expr(expr),
            StmtKind::Let { expr, .. } => self.visit_expr(expr),
            StmtKind::Expr(expr) => self.visit_expr(expr),
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        if let ExprKind::Call { func, args } = &expr.kind {
            self.visit_expr(func);
            for arg in args {
                self.visit_expr(arg);
            }
        }
    }
}
