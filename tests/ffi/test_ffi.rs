// tests/ffi/test_ffi.rs

#[test]
fn test_parse_source_ffi() {
    use std::ffi::CString;
    use crate::ffi::{taria_parse_source, TariaAstHandle};
    let src = CString::new("def foo(): pass").unwrap();
    let mut handle = TariaAstHandle { ptr: std::ptr::null_mut() };
    let result = unsafe { taria_parse_source(src.as_ptr(), &mut handle) };
    assert_eq!(result, 0);
    unsafe { crate::ffi::taria_free_ast(handle) };
}
// ... more FFI tests for error handling, ownership, etc.
