use nemotron_mlx::opencc::to_traditional;

#[test]
fn converts_simplified_to_taiwan_traditional_with_phrases() {
    assert_eq!(to_traditional("软件"), "軟體");
    assert_eq!(to_traditional("信息"), "資訊");
    assert_eq!(to_traditional("里面"), "裡面");
    assert_eq!(to_traditional("鼠标"), "滑鼠");
    assert_eq!(to_traditional("这是一个测试"), "這是一個測試");
}

#[test]
fn is_idempotent_on_traditional_and_passes_through_ascii() {
    assert_eq!(to_traditional("軟體與資訊"), "軟體與資訊");
    assert_eq!(to_traditional("hello, world 123"), "hello, world 123");
    assert_eq!(to_traditional(""), "");
}
