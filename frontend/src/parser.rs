//! Taria Parser: Recursive descent + Pratt for expressions

use crate::lexer::{Lexer, Token, TokenKind};
use crate::ast::*;

pub struct Parser<'src> {
    lexer: Lexer<'src>,
    current: Option<Token<'src>>,
}

impl<'src> Parser<'src> {
    pub fn new(src: &'src str) -> Self {
        let mut lexer = Lexer::new(src);
        let current = lexer.next_token();
        Parser { lexer, current }
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

    pub fn parse_module(&mut self) -> ModuleAST {
        let mut functions = Vec::new();

        while !self.current_is(TokenKind::Eof) {
            // A real parser would handle module-level statements and errors here
            if self.current_is(TokenKind::At) || self.current_is(TokenKind::Keyword) {
                functions.push(self.parse_function());
            } else {
                // Error recovery: bump if we don't know what it is at top level
                self.bump();
            }
        }

        ModuleAST { functions, span: 0..0 }
    }

    fn parse_function(&mut self) -> FunctionDecl {
        let start = self.current.as_ref().map(|t| t.span.start).unwrap_or(0);
        let mut decorators = Vec::new();

        while self.current_is(TokenKind::At) {
            decorators.push(self.parse_decorator());
        }

        // expect 'def' (Keyword)
        if self.current_is(TokenKind::Keyword) {
             self.bump(); // consume 'def'
        }

        let mut name = String::new();
        if self.current_is(TokenKind::Identifier) {
            if let Some(ref t) = self.current {
                name = t.slice.to_string();
            }
            self.bump();
        }

        // expect '('
        if self.current_is(TokenKind::LParen) { self.bump(); }

        // simple param parsing for demo
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
                if self.current_is(TokenKind::Identifier) {
                    if let Some(ref t) = self.current {
                        ty = Some(t.slice.to_string());
                    }
                    self.bump();
                }
            }
            params.push(Parameter { name: param_name, ty, span: 0..0 });

            if self.current_is(TokenKind::Comma) {
                self.bump();
            } else {
                break;
            }
        }

        // expect ')'
        if self.current_is(TokenKind::RParen) { self.bump(); }

        let mut return_type = None;
        if self.current_is(TokenKind::Arrow) {
            self.bump();
            if self.current_is(TokenKind::Identifier) {
                if let Some(ref t) = self.current {
                    return_type = Some(t.slice.to_string());
                }
                self.bump();
            }
        }

        // expect ':'
        if self.current_is(TokenKind::Colon) { self.bump(); }

        let mut body = Vec::new();
        // naive block parsing: just try to parse statements until dedent or end
        // for demo, we'll just parse one statement if it's a return

        if self.current_is(TokenKind::Newline) { self.bump(); }

        while self.current_is(TokenKind::Keyword) { // e.g. return
            if let Some(ref t) = self.current {
                if t.slice == "return" {
                    self.bump();
                    let expr = self.parse_expr(0);
                    body.push(Stmt { kind: StmtKind::Return(expr), span: 0..0 });
                    break; // break for demo
                } else {
                    self.bump(); // skip other keywords
                }
            }
        }

        let end = self.current.as_ref().map(|t| t.span.end).unwrap_or(0);

        FunctionDecl {
            name,
            decorators,
            params,
            return_type,
            body,
            span: start..end,
        }
    }

    fn parse_decorator(&mut self) -> Decorator {
        let start = self.current.as_ref().map(|t| t.span.start).unwrap_or(0);
        self.bump(); // consume '@'

        let mut name = String::new();
        if self.current_is(TokenKind::Identifier) {
            if let Some(ref t) = self.current {
                name = t.slice.to_string();
            }
            self.bump();
        }

        let mut args = Vec::new();
        if self.current_is(TokenKind::LParen) {
            self.bump();
            // skip to )
            while !self.current_is(TokenKind::RParen) && !self.current_is(TokenKind::Eof) {
                self.bump();
            }
            if self.current_is(TokenKind::RParen) { self.bump(); }
        }

        let end = self.current.as_ref().map(|t| t.span.end).unwrap_or(0);

        Decorator {
            name,
            args,
            span: start..end,
        }
    }

    // Pratt parser for expressions (stubbed)
    fn parse_expr(&mut self, _min_prec: u8) -> Expr {
        let start = self.current.as_ref().map(|t| t.span.start).unwrap_or(0);
        let mut kind = ExprKind::Literal("".into());

        if self.current_is(TokenKind::Identifier) {
            if let Some(ref t) = self.current {
                kind = ExprKind::Identifier(t.slice.to_string());
            }
            self.bump();
        }

        // ... implementation of pratt loop ...

        let end = self.current.as_ref().map(|t| t.span.end).unwrap_or(0);
        Expr { kind, span: start..end }
    }
}
