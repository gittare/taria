//! Taria Semantic Analysis: Type checking, Name resolution, and Semantic Validation
//!
//! This module performs pass 2 of the frontend compiler pipeline.

use crate::ast::*;
use crate::symbol_table::SymbolTable;
use crate::diagnostics::{DiagnosticsEngine, Diagnostic, Level};
use crate::source_map::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Void,
    Int8,
    Int32,
    Int64,
    Float32,
    Float64,
    /// Tensor(DataType, Dimensions)
    Tensor(Box<Type>, Vec<usize>),
    Unknown,
}

pub struct SemanticAnalyzer {
    pub sym_table: SymbolTable,
    pub diagnostics: DiagnosticsEngine,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            sym_table: SymbolTable::new(),
            diagnostics: DiagnosticsEngine::new(),
        }
    }

    pub fn analyze(&mut self, module: &ModuleAST) {
        for func in &module.functions {
            self.analyze_function(func);
        }
    }

    fn analyze_function(&mut self, func: &FunctionDecl) {
        self.sym_table.enter_scope();

        // Check decorators for GPU constraints
        let is_kernel = func.decorators.iter().any(|d| d.name == "gpu.kernel");
        if is_kernel {
            // Future: Validate block_size, shared_mem arguments via AST
        }

        // Register parameters
        for param in &func.params {
            let ty = self.parse_type(&param.ty);
            self.sym_table.insert(param.name.clone(), ty);
        }

        // Analyze body
        for stmt in &func.body {
            self.analyze_stmt(stmt);
        }

        self.sym_table.exit_scope();
    }

    fn analyze_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { name, expr } => {
                let ty = self.analyze_expr(expr);
                self.sym_table.insert(name.clone(), ty);
            }
            StmtKind::Return(expr) => {
                self.analyze_expr(expr);
                // Future: Verify against function return type
            }
            StmtKind::Expr(expr) => {
                self.analyze_expr(expr);
            }
        }
    }

    fn analyze_expr(&mut self, expr: &Expr) -> Type {
        match &expr.kind {
            ExprKind::Literal(_) => Type::Float32, // simplified
            ExprKind::Identifier(name) => {
                match self.sym_table.lookup(name) {
                    Some(ty) => ty,
                    None => {
                        self.diagnostics.emit(
                            Diagnostic::new(Level::Error, format!("Undefined variable: `{}`", name))
                            .with_span(expr.span)
                            .with_help("Verify the variable is declared before use.")
                        );
                        Type::Unknown
                    }
                }
            }
            ExprKind::Call { func, args } => {
                self.analyze_expr(func);
                for arg in args {
                    self.analyze_expr(arg);
                }
                Type::Tensor(Box::new(Type::Float32), vec![32, 32]) // Stub
            }
            ExprKind::Tensor(_) => {
                Type::Tensor(Box::new(Type::Float32), vec![])
            }
        }
    }

    fn parse_type(&self, ty_str: &Option<String>) -> Type {
        match ty_str {
            Some(s) if s.starts_with("Tensor") => Type::Tensor(Box::new(Type::Float32), vec![]), // basic approximation
            Some(s) if s == "CompressedChunk" => Type::Tensor(Box::new(Type::Int8), vec![]),
            _ => Type::Unknown,
        }
    }
}
