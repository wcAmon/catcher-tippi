use std::process::Command;

#[test]
fn help_describes_source_output_and_group_size_arguments() {
    let output = Command::new(env!("CARGO_BIN_EXE_nemotron-convert"))
        .arg("--help")
        .output()
        .expect("run converter help");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("--source"));
    assert!(stdout.contains("--output"));
    assert!(stdout.contains("--group-size"));
    assert!(stdout.contains("Nemotron 3.5"));
}

#[test]
fn numeric_group_size_never_panics_during_argument_parsing() {
    let output_path =
        std::env::temp_dir().join(format!("nemotron-convert-cli-test-{}", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_nemotron-convert"))
        .args([
            "--source",
            "missing.safetensors",
            "--output",
            output_path.to_str().unwrap(),
            "--group-size",
            "64",
        ])
        .output()
        .expect("run converter");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("conversion failed"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}
