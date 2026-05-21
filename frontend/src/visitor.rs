//! AST Visitor pattern

use crate::ast::*;

pub trait Visitor<'ast> {
    fn visit_module(&mut self, module: &'ast ModuleAST) {
        walk_module(self, module);
    }

    fn visit_function(&mut self, func: &'ast FunctionDecl) {
        walk_function(self, func);
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        walk_expr(self, expr);
    }
}

pub fn walk_module<'ast, V: Visitor<'ast> + ?Sized>(visitor: &mut V, module: &'ast ModuleAST) {
    for func in &module.functions {
        visitor.visit_function(func);
    }
}

pub fn walk_function<'ast, V: Visitor<'ast> + ?Sized>(visitor: &mut V, func: &'ast FunctionDecl) {
    for stmt in &func.body {
        visitor.visit_stmt(stmt);
    }
}

pub fn walk_stmt<'ast, V: Visitor<'ast> + ?Sized>(visitor: &mut V, stmt: &'ast Stmt) {
    match &stmt.kind {
        StmtKind::Let { expr, .. } => visitor.visit_expr(expr),
        StmtKind::Return(expr) => visitor.visit_expr(expr),
        StmtKind::Expr(expr) => visitor.visit_expr(expr),
    }
}

pub fn walk_expr<'ast, V: Visitor<'ast> + ?Sized>(visitor: &mut V, expr: &'ast Expr) {
    match &expr.kind {
        ExprKind::Call { func, args } => {
            visitor.visit_expr(func);
            for arg in args {
                visitor.visit_expr(arg);
            }
        }
        _ => {}
    }
}
