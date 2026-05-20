// tests/frontend/test_parser.rs

#[test]
fn test_parse_function_with_decorator() {
    let src = "@gpu.kernel\ndef compress(x: Tensor) -> Latent:\n    return encoder(x)";
    let mut parser = crate::parser::Parser::new(src);
    let module = parser.parse_module();
    assert_eq!(module.functions.len(), 1);
    let func = &module.functions[0];
    assert_eq!(func.name, "compress");
    assert_eq!(func.decorators.len(), 1);
}
// ... more tests for error recovery, type parsing, etc.
