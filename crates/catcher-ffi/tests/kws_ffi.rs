use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;

use catcher_ffi::{
    CATCHER_COMMAND_DETECTED, CATCHER_INVALID_ARGUMENT, CATCHER_NO_UPDATE, CATCHER_OK,
    catcher_kws_create, catcher_kws_destroy, catcher_kws_keyword, catcher_kws_last_error,
    catcher_kws_push_audio, catcher_kws_start, catcher_kws_start_ms,
};

fn read_wav(path: impl AsRef<Path>) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate, 16_000);
    assert_eq!(spec.bits_per_sample, 16);
    reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32_768.0)
        .collect()
}

unsafe fn feed_until_detected(handle: *mut catcher_ffi::KwsHandle, samples: &[f32]) -> bool {
    for chunk in samples.chunks(1_600) {
        let status = unsafe { catcher_kws_push_audio(handle, chunk.as_ptr(), chunk.len()) };
        assert!(
            status == CATCHER_NO_UPDATE || status == CATCHER_COMMAND_DETECTED,
            "unexpected KWS push status {status}"
        );
        if status == CATCHER_COMMAND_DETECTED {
            return true;
        }
    }
    false
}

unsafe fn assert_no_detection(handle: *mut catcher_ffi::KwsHandle, samples: &[f32]) {
    for chunk in samples.chunks(1_600) {
        assert_eq!(
            unsafe { catcher_kws_push_audio(handle, chunk.as_ptr(), chunk.len()) },
            CATCHER_NO_UPDATE
        );
    }
}

fn padded_fixture(name: &str) -> Vec<f32> {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    let spoken = read_wav(path);
    let mut samples = vec![0.0; 16_000];
    samples.extend(spoken);
    samples.extend(vec![0.0; 16_000]);
    samples
}

#[test]
fn null_kws_handle_is_safe() {
    unsafe {
        assert!(catcher_kws_create(ptr::null()).is_null());
        assert_eq!(catcher_kws_start(ptr::null_mut()), CATCHER_INVALID_ARGUMENT);
        assert_eq!(
            catcher_kws_push_audio(ptr::null_mut(), ptr::null(), 0),
            CATCHER_INVALID_ARGUMENT
        );
        assert!(catcher_kws_keyword(ptr::null()).is_null());
        assert_eq!(catcher_kws_start_ms(ptr::null()), 0);
        assert!(!catcher_kws_last_error(ptr::null()).is_null());
        catcher_kws_destroy(ptr::null_mut());
    }
}

#[test]
#[ignore = "requires SHERPA_KWS_MODEL"]
fn detects_both_mandarin_submit_fixtures_and_resets() {
    let model = CString::new(std::env::var("SHERPA_KWS_MODEL").unwrap()).unwrap();
    let handle = unsafe { catcher_kws_create(model.as_ptr()) };
    assert!(!handle.is_null(), "KWS model should load");

    unsafe {
        for fixture in ["bang-wo-song-chu-zh-cn.wav", "bang-wo-song-chu-zh-tw.wav"] {
            let samples = padded_fixture(fixture);
            assert_eq!(catcher_kws_start(handle), CATCHER_OK);
            assert!(feed_until_detected(handle, &samples), "missed {fixture}");
            assert_eq!(
                CStr::from_ptr(catcher_kws_keyword(handle))
                    .to_str()
                    .unwrap(),
                "SUBMIT_ZH"
            );
            let start_ms = catcher_kws_start_ms(handle);
            assert!((500..=3_000).contains(&start_ms), "{fixture}: {start_ms}ms");
            assert_eq!(
                catcher_kws_push_audio(handle, ptr::null(), 0),
                CATCHER_NO_UPDATE
            );
            assert_eq!(
                CStr::from_ptr(catcher_kws_keyword(handle))
                    .to_str()
                    .unwrap(),
                "SUBMIT_ZH"
            );
            assert_eq!(catcher_kws_start_ms(handle), start_ms);

            assert_eq!(catcher_kws_start(handle), CATCHER_OK);
            assert_eq!(CStr::from_ptr(catcher_kws_keyword(handle)).to_bytes(), b"");
            assert_eq!(catcher_kws_start_ms(handle), 0);
            assert!(
                feed_until_detected(handle, &samples),
                "missed {fixture} after reset"
            );
        }
        catcher_kws_destroy(handle);
    }
}

#[test]
#[ignore = "requires SHERPA_KWS_MODEL"]
fn partial_old_and_unrelated_audio_do_not_trigger_submit() {
    let model = CString::new(std::env::var("SHERPA_KWS_MODEL").unwrap()).unwrap();
    let handle = unsafe { catcher_kws_create(model.as_ptr()) };
    assert!(!handle.is_null(), "KWS model should load");

    unsafe {
        for fixture in [
            "bang-wo-zh-tw.wav",
            "song-chu-zh-tw.wav",
            "tippi-go.wav",
            "hello-streaming.wav",
            "conversation.wav",
        ] {
            assert_eq!(catcher_kws_start(handle), CATCHER_OK);
            assert_no_detection(handle, &padded_fixture(fixture));
        }
        catcher_kws_destroy(handle);
    }
}

#[test]
#[ignore = "requires SHERPA_KWS_MODEL"]
fn reported_timestamp_is_not_an_absolute_long_stream_offset() {
    let model = CString::new(std::env::var("SHERPA_KWS_MODEL").unwrap()).unwrap();
    let handle = unsafe { catcher_kws_create(model.as_ptr()) };
    assert!(!handle.is_null());
    let spoken = read_wav(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/bang-wo-song-chu-zh-tw.wav"
    ));

    unsafe {
        for prefix_seconds in [1_usize, 5, 10, 20] {
            assert_eq!(catcher_kws_start(handle), CATCHER_OK);
            let mut samples = vec![0.0; prefix_seconds * 16_000];
            samples.extend_from_slice(&spoken);
            samples.extend(vec![0.0; 16_000]);
            assert!(feed_until_detected(handle, &samples));
            let reported = catcher_kws_start_ms(handle);
            println!("{prefix_seconds}s prefix reported {reported}ms");
            assert!(
                reported <= 3_000,
                "{prefix_seconds}s prefix unexpectedly produced absolute {reported}ms"
            );
        }
        catcher_kws_destroy(handle);
    }
}
