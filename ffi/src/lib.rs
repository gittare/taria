//! Taria FFI: ABI-stable, panic-safe, thread-safe bindings for the Rust frontend

use std::os::raw::{c_char, c_void};
use std::ffi::CStr;
use std::panic::catch_unwind;

use frontend::ast::ModuleAST;
use frontend::parser::Parser;

/// Opaque handle to the parsed AST
#[repr(C)]
pub struct TariaAstHandle {
    pub ptr: *mut c_void, // Opaque pointer to Box<ModuleAST>
}

/// Parses the given Taria source code string into an AST.
/// Returns 0 on success, or a negative error code on failure.
#[no_mangle]
pub extern "C" fn taria_parse_source(
    src: *const c_char,
    out_handle: *mut TariaAstHandle,
) -> i32 {
    if src.is_null() || out_handle.is_null() {
        return -1;
    }

    let result = catch_unwind(|| {
        let c_str = unsafe { CStr::from_ptr(src) };
        let src_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return -2, // Invalid UTF-8
        };

        let mut parser = Parser::new(src_str);
        let ast = parser.parse_module();
        let boxed_ast = Box::new(ast);
        let ptr = Box::into_raw(boxed_ast) as *mut c_void;

        unsafe {
            (*out_handle).ptr = ptr;
        }

        0 // Success
    });

    match result {
        Ok(code) => code,
        Err(_) => -3, // Panic occurred
    }
}

/// Frees the AST handle previously allocated by `taria_parse_source`.
#[no_mangle]
pub extern "C" fn taria_free_ast(handle: TariaAstHandle) {
    if !handle.ptr.is_null() {
        let _ = catch_unwind(|| {
            unsafe {
                let _ = Box::from_raw(handle.ptr as *mut ModuleAST);
            }
        });
    }
}
