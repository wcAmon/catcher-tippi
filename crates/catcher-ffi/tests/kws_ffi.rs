use std::ffi::{CStr, CString};
use std::path::Path;
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
fn detects_padded_tippi_go_once_and_resets() {
    let model = CString::new(std::env::var("SHERPA_KWS_MODEL").unwrap()).unwrap();
    let handle = unsafe { catcher_kws_create(model.as_ptr()) };
    assert!(!handle.is_null(), "KWS model should load");
    let spoken = read_wav(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tippi-go.wav"
    ));
    let mut samples = vec![0.0; 16_000];
    samples.extend(spoken);
    samples.extend(vec![0.0; 16_000]);

    unsafe {
        assert_eq!(catcher_kws_start(handle), CATCHER_OK);
        assert!(feed_until_detected(handle, &samples));
        assert_eq!(
            CStr::from_ptr(catcher_kws_keyword(handle))
                .to_str()
                .unwrap(),
            "TIPPI_GO"
        );
        let start_ms = catcher_kws_start_ms(handle);
        assert!((800..=2_000).contains(&start_ms), "start_ms was {start_ms}");
        println!("detected TIPPI_GO at {start_ms} ms");

        assert_eq!(
            catcher_kws_push_audio(handle, ptr::null(), 0),
            CATCHER_NO_UPDATE
        );
        assert_eq!(
            CStr::from_ptr(catcher_kws_keyword(handle))
                .to_str()
                .unwrap(),
            "TIPPI_GO"
        );
        assert_eq!(catcher_kws_start_ms(handle), start_ms);

        assert_eq!(catcher_kws_start(handle), CATCHER_OK);
        assert_eq!(CStr::from_ptr(catcher_kws_keyword(handle)).to_bytes(), b"");
        assert_eq!(catcher_kws_start_ms(handle), 0);
        assert!(feed_until_detected(handle, &samples));
        catcher_kws_destroy(handle);
    }
}

#[test]
#[ignore = "requires SHERPA_KWS_MODEL"]
fn unrelated_audio_does_not_trigger_tippi_go() {
    let model = CString::new(std::env::var("SHERPA_KWS_MODEL").unwrap()).unwrap();
    let handle = unsafe { catcher_kws_create(model.as_ptr()) };
    assert!(!handle.is_null(), "KWS model should load");

    let hello = read_wav(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hello-streaming.wav"
    ));
    let conversation = read_wav(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/conversation.wav"
    ));
    let midpoint = conversation.len() / 2;

    unsafe {
        assert_eq!(catcher_kws_start(handle), CATCHER_OK);
        assert_no_detection(handle, &hello);
        assert_eq!(catcher_kws_start(handle), CATCHER_OK);
        assert_no_detection(handle, &conversation[..midpoint]);
        assert_eq!(catcher_kws_start(handle), CATCHER_OK);
        assert_no_detection(handle, &conversation[midpoint..]);
        catcher_kws_destroy(handle);
    }
}
