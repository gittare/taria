//! Taria Compiler Diagnostics System
//! Provides rustc/clang style colored error reporting with spans and suggestions.

use crate::source_map::{SourceFile, Span};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Error,
    Warning,
    Note,
    Help,
}

impl Level {
    pub fn to_str(&self) -> &'static str {
        match self {
            Level::Error => "error",
            Level::Warning => "warning",
            Level::Note => "note",
            Level::Help => "help",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: Level,
    pub message: String,
    pub span: Option<Span>,
    pub help: Option<String>,
}

impl Diagnostic {
    pub fn new(level: Level, message: impl Into<String>) -> Self {
        Diagnostic {
            level,
            message: message.into(),
            span: None,
            help: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
}

pub struct DiagnosticsEngine {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticsEngine {
    pub fn new() -> Self {
        DiagnosticsEngine {
            diagnostics: Vec::new(),
        }
    }

    pub fn emit(&mut self, diag: Diagnostic) {
        self.diagnostics.push(diag);
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.level == Level::Error)
    }

    pub fn print(&self, source_file: &SourceFile) {
        for diag in &self.diagnostics {
            let prefix = diag.level.to_str();

            // Simplified terminal output logic.
            // In a production setup, we would use termcolor and calculate line/column numbers.
            println!("{}: {}", prefix, diag.message);

            if let Some(span) = diag.span {
                let snippet = source_file.get_snippet(span);
                println!("  --> {}:{}:{}", source_file.name, span.lo, span.hi);
                println!("   | ");
                println!("   | {}", snippet.replace("\n", "\n   | "));
                println!("   | ");
            }

            if let Some(ref help) = diag.help {
                println!("help: {}", help);
            }

            println!("");
        }
    }
}
