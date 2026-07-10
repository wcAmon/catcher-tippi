use std::fs;

use nemotron_mlx::tokenizer::Tokenizer;

#[test]
fn decodes_bpe_metaspace_and_strips_model_control_ids() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("tokenizer.json");
    fs::write(
        &path,
        r#"{
          "model": {"type":"BPE", "vocab":{"▁hello":0,"世":1,"界":2}},
          "added_tokens": [
            {"id":3,"content":"<en-US>","special":true},
            {"id":4,"content":"<pad>","special":true},
            {"id":5,"content":"<blank>","special":true}
          ],
          "decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}
        }"#,
    )
    .unwrap();
    let tokenizer = Tokenizer::from_json(&path, 4, 5).unwrap();

    assert_eq!(
        tokenizer.decode(&[3, 0, 1, 2, 4, 5], true).unwrap(),
        "hello世界"
    );
    assert_eq!(
        tokenizer.decode(&[3, 0, 1, 2, 4, 5], false).unwrap(),
        "<en-US> hello世界"
    );
    assert!(tokenizer.decode(&[99], true).is_err());
}
