// frontend/src/lexer.rs

//! Taria Lexer: Zero-copy, SIMD-aware, Pythonic + Rust-safe

use std::str::CharIndices;
use std::ops::Range;

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
    // ... extensible for future tokens
}

#[derive(Debug, Clone)]
pub struct Token<'src> {
    pub kind: TokenKind,
    pub span: Range<usize>, // Byte offsets in source
    pub slice: &'src str,   // Zero-copy view
}

pub struct Lexer<'src> {
    src: &'src str,
    chars: CharIndices<'src>,
    pos: usize,
    indent_stack: Vec<usize>,
    // ... arena allocator for tokens (see bumpalo or typed-arena)
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Lexer {
            src,
            chars: src.char_indices(),
            pos: 0,
            indent_stack: vec![0],
        }
    }

    /// Main lexing loop: yields tokens with spans and zero-copy slices.
    pub fn next_token(&mut self) -> Option<Token<'src>> {
        // SIMD-aware scanning: use memchr/memchr2 for fast newline/space detection
        // (see memchr crate for real impl)
        // ... implementation omitted for brevity ...
        None
    }

    // Helper: scan identifier, decorator, literal, etc.
    // ... implementation details ...
}

// Arena allocation: Use bumpalo::Bump or typed-arena for fast, zero-GC token storage.
// SIMD: Use memchr for fast newline/indent detection.
// Span tracking: All tokens carry byte offsets for diagnostics.
