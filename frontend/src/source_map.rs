//! Source Map for span tracking and diagnostics.

use std::ops::Range;
use std::rc::Rc;

pub type BytePos = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub lo: BytePos,
    pub hi: BytePos,
}

impl Span {
    pub fn new(lo: BytePos, hi: BytePos) -> Self {
        Span { lo, hi }
    }

    pub fn merge(&self, other: &Span) -> Span {
        Span {
            lo: std::cmp::min(self.lo, other.lo),
            hi: std::cmp::max(self.hi, other.hi),
        }
    }
}

impl Default for Span {
    fn default() -> Self {
        Span { lo: 0, hi: 0 }
    }
}

impl From<Range<usize>> for Span {
    fn from(r: Range<usize>) -> Self {
        Span { lo: r.start, hi: r.end }
    }
}

impl Into<Range<usize>> for Span {
    fn into(self) -> Range<usize> {
        self.lo..self.hi
    }
}

/// Represents a source file loaded into the compiler.
#[derive(Debug)]
pub struct SourceFile {
    pub name: String,
    pub src: Rc<String>,
}

impl SourceFile {
    pub fn new(name: String, src: String) -> Self {
        SourceFile {
            name,
            src: Rc::new(src),
        }
    }

    /// Extracts a string slice for a given span.
    pub fn get_snippet(&self, span: Span) -> &str {
        if span.lo <= span.hi && span.hi <= self.src.len() {
            &self.src[span.lo..span.hi]
        } else {
            ""
        }
    }
}
