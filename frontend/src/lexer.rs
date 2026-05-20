//! Taria Lexer: Zero-copy, SIMD-aware, Pythonic + Rust-safe

use std::str::CharIndices;
use std::ops::Range;

/// The kind of token
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Decorators: @gpu.kernel, @compress, etc.
    At,
    Identifier,
    Tensor,
    Attribute,
    Literal,
    Indent,
    Dedent,
    Newline,
    Keyword,
    Operator,
    Colon,
    Comma,
    LParen,
    RParen,
    Arrow,      // ->
    Annotation, // :Type
    Eof,
    Error,
}

/// A lexical token emitted by the lexer
#[derive(Debug, Clone)]
pub struct Token<'src> {
    pub kind: TokenKind,
    pub span: Range<usize>, // Byte offsets in source
    pub slice: &'src str,   // Zero-copy view
}

/// The lexer
pub struct Lexer<'src> {
    src: &'src str,
    pos: usize,
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

    /// Helper to get the remaining string
    fn remaining(&self) -> &'src str {
        &self.src[self.pos..]
    }

    /// Advance lexer by N bytes
    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    /// Skip whitespaces (SIMD-aware scanning would be implemented here in production using `memchr`)
    fn skip_whitespace(&mut self) {
        let mut chars = self.remaining().char_indices();
        while let Some((_, c)) = chars.next() {
            if c != ' ' && c != '\t' && c != '\r' {
                break;
            }
            self.advance(c.len_utf8());
        }
    }

    /// Main lexing loop: yields tokens with spans and zero-copy slices.
    pub fn next_token(&mut self) -> Option<Token<'src>> {
        self.skip_whitespace();

        if self.pos >= self.src.len() {
            return Some(Token {
                kind: TokenKind::Eof,
                span: self.pos..self.pos,
                slice: "",
            });
        }

        let remaining = self.remaining();
        let c = remaining.chars().next().unwrap();
        let start = self.pos;
        let c_len = c.len_utf8();

        // Simple match for demonstration
        let kind = match c {
            '@' => TokenKind::At,
            ':' => TokenKind::Colon,
            ',' => TokenKind::Comma,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '-' if remaining.starts_with("->") => {
                self.advance(2);
                return Some(Token {
                    kind: TokenKind::Arrow,
                    span: start..self.pos,
                    slice: &self.src[start..self.pos],
                });
            }
            '\n' => TokenKind::Newline,
            _ if c.is_alphabetic() || c == '_' => {
                return Some(self.scan_identifier());
            }
            _ if c.is_numeric() => {
                return Some(self.scan_number());
            }
            _ => TokenKind::Error, // Fallback
        };

        if kind != TokenKind::Error {
            self.advance(c_len);
        } else {
            // consume one char on error
            self.advance(c_len);
        }

        Some(Token {
            kind,
            span: start..self.pos,
            slice: &self.src[start..self.pos],
        })
    }

    fn scan_identifier(&mut self) -> Token<'src> {
        let start = self.pos;
        let mut chars = self.remaining().char_indices();
        while let Some((_, c)) = chars.next() {
            if !c.is_alphanumeric() && c != '_' && c != '.' {
                break;
            }
            self.advance(c.len_utf8());
        }

        let slice = &self.src[start..self.pos];
        let kind = match slice {
            "def" | "return" | "if" | "else" => TokenKind::Keyword,
            _ => TokenKind::Identifier,
        };

        Token {
            kind,
            span: start..self.pos,
            slice,
        }
    }

    fn scan_number(&mut self) -> Token<'src> {
        let start = self.pos;
        let mut chars = self.remaining().char_indices();
        while let Some((_, c)) = chars.next() {
            if !c.is_numeric() && c != '.' {
                break;
            }
            self.advance(c.len_utf8());
        }

        Token {
            kind: TokenKind::Literal,
            span: start..self.pos,
            slice: &self.src[start..self.pos],
        }
    }
}
