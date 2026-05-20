//! Taria FFI: ABI-stable, panic-safe, thread-safe

use std::os::raw::{c_char, c_void};
use std::ffi::{CStr, CString};
use std::sync::Arc;

#[repr(C)]
pub struct TariaAstHandle {
    ptr: *mut c_void, // Opaque pointer to AST
}

#[no_mangle]
pub extern "C" fn taria_parse_source(
    src: *const c_char,
    out_handle: *mut TariaAstHandle,
) -> i32 {
    // Safety: check null, catch panics, return error codes
    // ... parse source, allocate AST, store in handle ...
    0 // success
}

#[no_mangle]
pub extern "C" fn taria_free_ast(handle: TariaAstHandle) {
    // Free AST, handle ownership
}

// ABI: #[repr(C)], no generics, no Rust-specific types
// Panic isolation: catch_unwind
// Thread safety: Arc for shared data, Send/Sync
// Serialization: Flatbuffers/Cap'n Proto for cross-lang AST
