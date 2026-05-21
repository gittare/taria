//! Taria Lexer: Zero-copy, SIMD-aware, Pythonic + Rust-safe

use std::ops::Range;

use crate::token::{Token, TokenKind};
use crate::source_map::Span;

/// The lexer state machine
pub struct Lexer<'src> {
    src: &'src str,
    pos: usize,
    // Indentation stack for Python-like semantic blocks
    indent_stack: Vec<usize>,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Lexer {
            src,
            pos: 0,
            indent_stack: vec![0],
        }
    }

    fn remaining(&self) -> &'src str {
        &self.src[self.pos..]
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    /// Skips spaces (but not newlines, since we need to track indents eventually).
    /// In a production system, this uses memchr for SIMD-accelerated scanning.
    fn skip_whitespace(&mut self) {
        let mut chars = self.remaining().char_indices();
        while let Some((_, c)) = chars.next() {
            if c == ' ' || c == '\t' || c == '\r' {
                self.advance(c.len_utf8());
            } else {
                break;
            }
        }
    }

    /// Main lexing loop: yields tokens with spans and zero-copy slices.
    pub fn next_token(&mut self) -> Option<Token<'src>> {
        self.skip_whitespace();

        if self.pos >= self.src.len() {
            return Some(Token {
                kind: TokenKind::Eof,
                span: Span::new(self.pos, self.pos),
                slice: "",
            });
        }

        let remaining = self.remaining();
        let c = remaining.chars().next().unwrap();
        let start = self.pos;
        let c_len = c.len_utf8();

        let kind = match c {
            '@' => TokenKind::At,
            ':' => TokenKind::Colon,
            ',' => TokenKind::Comma,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            '=' => TokenKind::Assign,
            '+' => TokenKind::Plus,
            '-' => {
                if remaining.starts_with("->") {
                    self.advance(2);
                    return Some(Token {
                        kind: TokenKind::Arrow,
                        span: Span::new(start, self.pos),
                        slice: &self.src[start..self.pos],
                    });
                } else {
                    TokenKind::Minus
                }
            }
            '*' => TokenKind::Star,
            '/' => TokenKind::Slash,
            '.' => TokenKind::Dot,
            '\n' => TokenKind::Newline,
            _ if c.is_alphabetic() || c == '_' => {
                return Some(self.scan_identifier());
            }
            _ if c.is_numeric() => {
                return Some(self.scan_number());
            }
            _ => TokenKind::Error,
        };

        self.advance(c_len);

        Some(Token {
            kind,
            span: Span::new(start, self.pos),
            slice: &self.src[start..self.pos],
        })
    }

    fn scan_identifier(&mut self) -> Token<'src> {
        let start = self.pos;
        let mut chars = self.remaining().char_indices();
        while let Some((_, c)) = chars.next() {
            if !c.is_alphanumeric() && c != '_' {
                break;
            }
            self.advance(c.len_utf8());
        }

        let slice = &self.src[start..self.pos];
        let kind = match slice {
            "def" => TokenKind::Def,
            "return" => TokenKind::Return,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "for" => TokenKind::For,
            "while" => TokenKind::While,
            "Tensor" => TokenKind::Tensor,
            _ => TokenKind::Identifier,
        };

        Token {
            kind,
            span: Span::new(start, self.pos),
            slice,
        }
    }

    fn scan_number(&mut self) -> Token<'src> {
        let start = self.pos;
        let mut chars = self.remaining().char_indices();
        let mut is_float = false;

        while let Some((_, c)) = chars.next() {
            if c == '.' {
                is_float = true;
                self.advance(c.len_utf8());
            } else if c.is_numeric() {
                self.advance(c.len_utf8());
            } else {
                break;
            }
        }

        Token {
            kind: if is_float { TokenKind::Float } else { TokenKind::Integer },
            span: Span::new(start, self.pos),
            slice: &self.src[start..self.pos],
        }
    }
}
