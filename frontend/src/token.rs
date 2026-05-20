//! Token definitions for the Taria lexer.

use crate::source_map::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Keywords
    Def,
    Return,
    If,
    Else,
    For,
    While,

    // Decorators & Attributes
    At,          // @
    Attribute,   // e.g., gpu.kernel

    // Identifiers & Literals
    Identifier,
    Integer,
    Float,
    String,
    Tensor,      // Built-in Tensor type keyword

    // Punctuation & Operators
    Colon,
    Comma,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Arrow,       // ->
    Assign,      // =
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Dot,         // .

    // Structural
    Indent,
    Dedent,
    Newline,
    Eof,
    Error,
}

#[derive(Debug, Clone)]
pub struct Token<'src> {
    pub kind: TokenKind,
    pub span: Span,
    pub slice: &'src str,
}
