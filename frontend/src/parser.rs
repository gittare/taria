//! Taria Parser: Recursive descent + Pratt for expressions

use crate::lexer::Lexer;
use crate::token::{Token, TokenKind};
use crate::ast::*;
use crate::source_map::Span;
use crate::diagnostics::{DiagnosticsEngine, Diagnostic, Level};

pub struct Parser<'src> {
    lexer: Lexer<'src>,
    current: Option<Token<'src>>,
    pub diagnostics: DiagnosticsEngine,
}

impl<'src> Parser<'src> {
    pub fn new(src: &'src str) -> Self {
        let mut lexer = Lexer::new(src);
        let current = lexer.next_token();
        Parser {
            lexer,
            current,
            diagnostics: DiagnosticsEngine::new()
        }
    }

    fn bump(&mut self) {
        self.current = self.lexer.next_token();
    }

    fn current_is(&self, kind: TokenKind) -> bool {
        if let Some(ref token) = self.current {
            token.kind == kind
        } else {
            false
        }
    }

    fn span(&self) -> Span {
        self.current.as_ref().map(|t| t.span).unwrap_or(Span::default())
    }

    fn expect(&mut self, kind: TokenKind) -> bool {
        if self.current_is(kind.clone()) {
            self.bump();
            true
        } else {
            let span = self.span();
            self.diagnostics.emit(
                Diagnostic::new(Level::Error, format!("Expected {:?}", kind))
                .with_span(span)
                .with_help("Check syntax")
            );
            false
        }
    }

    pub fn parse_module(&mut self) -> ModuleAST {
        let mut functions = Vec::new();
        let start = self.span().lo;

        while !self.current_is(TokenKind::Eof) {
            if self.current_is(TokenKind::At) || self.current_is(TokenKind::Def) {
                functions.push(self.parse_function());
            } else {
                // Skip unknown tokens at top level
                self.bump();
            }
        }

        let end = self.span().hi;
        ModuleAST { functions, span: Span::new(start, end) }
    }

    fn parse_function(&mut self) -> FunctionDecl {
        let start = self.span().lo;
        let mut decorators = Vec::new();

        while self.current_is(TokenKind::At) {
            decorators.push(self.parse_decorator());
        }

        self.expect(TokenKind::Def);

        let mut name = String::new();
        if self.current_is(TokenKind::Identifier) {
            if let Some(ref t) = self.current {
                name = t.slice.to_string();
            }
            self.bump();
        }

        self.expect(TokenKind::LParen);

        let mut params = Vec::new();
        while self.current_is(TokenKind::Identifier) {
            let mut param_name = String::new();
            if let Some(ref t) = self.current {
                param_name = t.slice.to_string();
            }
            self.bump();

            let mut ty = None;
            if self.current_is(TokenKind::Colon) {
                self.bump();
                if self.current_is(TokenKind::Identifier) || self.current_is(TokenKind::Tensor) {
                    if let Some(ref t) = self.current {
                        ty = Some(t.slice.to_string());
                    }
                    self.bump();
                }
            }
            params.push(Parameter { name: param_name, ty, span: Span::default() });

            if self.current_is(TokenKind::Comma) {
                self.bump();
            } else {
                break;
            }
        }

        self.expect(TokenKind::RParen);

        let mut return_type = None;
        if self.current_is(TokenKind::Arrow) {
            self.bump();
            if self.current_is(TokenKind::Identifier) || self.current_is(TokenKind::Tensor) {
                if let Some(ref t) = self.current {
                    return_type = Some(t.slice.to_string());
                }
                self.bump();
            }
        }

        self.expect(TokenKind::Colon);

        let mut body = Vec::new();

        if self.current_is(TokenKind::Newline) { self.bump(); }

        while !self.current_is(TokenKind::Eof) && !self.current_is(TokenKind::Def) && !self.current_is(TokenKind::At) {
            if self.current_is(TokenKind::Return) {
                self.bump();
                let expr = self.parse_expr(0);
                body.push(Stmt { kind: StmtKind::Return(expr), span: Span::default() });
            } else if self.current_is(TokenKind::Identifier) {
                // Very basic parsing for assignment or expr
                let name = self.current.as_ref().unwrap().slice.to_string();
                self.bump();
                if self.current_is(TokenKind::Assign) {
                    self.bump();
                    let expr = self.parse_expr(0);
                    body.push(Stmt { kind: StmtKind::Let { name, expr }, span: Span::default() });
                } else {
                    let expr = Expr { kind: ExprKind::Identifier(name), span: Span::default() };
                    body.push(Stmt { kind: StmtKind::Expr(expr), span: Span::default() });
                }
            } else {
                self.bump();
            }
        }

        let end = self.span().hi;

        FunctionDecl {
            name,
            decorators,
            params,
            return_type,
            body,
            span: Span::new(start, end),
        }
    }

    fn parse_decorator(&mut self) -> Decorator {
        let start = self.span().lo;
        self.bump(); // consume '@'

        let mut name = String::new();
        // Decorator name might be complex e.g. gpu.kernel
        while self.current_is(TokenKind::Identifier) || self.current_is(TokenKind::Dot) {
            if let Some(ref t) = self.current {
                name.push_str(t.slice);
            }
            self.bump();
        }

        let mut args = Vec::new();
        if self.current_is(TokenKind::LParen) {
            self.bump();
            while !self.current_is(TokenKind::RParen) && !self.current_is(TokenKind::Eof) {
                self.bump(); // simplistic skipping for args
            }
            if self.current_is(TokenKind::RParen) { self.bump(); }
        }

        let end = self.span().hi;

        Decorator {
            name,
            args,
            span: Span::new(start, end),
        }
    }

    fn parse_expr(&mut self, _min_prec: u8) -> Expr {
        let start = self.span().lo;
        let mut kind = ExprKind::Literal("".into());

        if self.current_is(TokenKind::Identifier) {
            if let Some(ref t) = self.current {
                kind = ExprKind::Identifier(t.slice.to_string());
            }
            self.bump();

            // Check for Call
            if self.current_is(TokenKind::LParen) {
                self.bump();
                let mut args = Vec::new();
                if self.current_is(TokenKind::Identifier) {
                    let name = self.current.as_ref().unwrap().slice.to_string();
                    args.push(Expr { kind: ExprKind::Identifier(name), span: Span::default() });
                    self.bump();
                }
                if self.current_is(TokenKind::RParen) { self.bump(); }

                let func_expr = Expr { kind: kind.clone(), span: Span::default() };
                kind = ExprKind::Call { func: Box::new(func_expr), args };
            }
        } else if self.current_is(TokenKind::Integer) || self.current_is(TokenKind::Float) {
            if let Some(ref t) = self.current {
                kind = ExprKind::Literal(t.slice.to_string());
            }
            self.bump();
        }

        let end = self.span().hi;
        Expr { kind, span: Span::new(start, end) }
    }
}
