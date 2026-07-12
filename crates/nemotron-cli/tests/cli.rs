use std::process::Command;

#[test]
fn help_exposes_streaming_transcription_arguments() {
    let output = Command::new(env!("CARGO_BIN_EXE_catcher"))
        .args(["transcribe", "--help"])
        .output()
        .expect("run CLI help");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for argument in [
        "--model",
        "--audio",
        "--language",
        "--lookahead",
        "--tokenizer",
    ] {
        assert!(stdout.contains(argument), "missing {argument} in {stdout}");
    }
}

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT"]
fn transcribes_the_reference_wav_end_to_end() {
    let model = std::env::var("NEMOTRON_MLX_ARTIFACT").expect("artifact path");
    let audio = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hello-streaming.wav"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_catcher"))
        .args([
            "transcribe",
            "--model",
            &model,
            "--audio",
            audio,
            "--language",
            "en-US",
            "--lookahead",
            "3",
        ])
        .output()
        .expect("run transcription CLI");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        "Hello, this is a streaming speech recognition test"
    );
}

#[test]
fn help_lists_the_diarize_subcommand() {
    let output = Command::new(env!("CARGO_BIN_EXE_catcher"))
        .arg("--help")
        .output()
        .expect("run catcher help");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("diarize"));
}

#[test]
fn diarize_with_missing_model_fails_cleanly() {
    let output = Command::new(env!("CARGO_BIN_EXE_catcher"))
        .args([
            "diarize",
            "--model",
            "/nonexistent",
            "--audio",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/fixtures/hello-streaming.wav"
            ),
        ])
        .output()
        .expect("run catcher diarize");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("catcher:"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}
