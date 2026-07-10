use std::process::Command;

#[test]
fn help_describes_source_and_output_arguments() {
    let output = Command::new(env!("CARGO_BIN_EXE_nemotron-convert"))
        .arg("--help")
        .output()
        .expect("run converter help");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("--source"));
    assert!(stdout.contains("--output"));
    assert!(stdout.contains("Nemotron 3.5"));
}
