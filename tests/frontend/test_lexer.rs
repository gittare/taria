// tests/frontend/test_lexer.rs

#[test]
fn test_decorator_token() {
    let src = "@gpu.kernel";
    let mut lexer = crate::lexer::Lexer::new(src);
    let token = lexer.next_token().unwrap();
    assert_eq!(token.kind, crate::lexer::TokenKind::At);
}

#[test]
fn test_identifier_token() {
    let src = "compress";
    let mut lexer = crate::lexer::Lexer::new(src);
    let token = lexer.next_token().unwrap();
    assert_eq!(token.kind, crate::lexer::TokenKind::Identifier);
}
// ... more tests for literals, tensors, indentation, etc.
