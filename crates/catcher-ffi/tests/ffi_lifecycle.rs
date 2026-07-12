use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use catcher_ffi::{
    CATCHER_INVALID_ARGUMENT, CATCHER_INVALID_STATE, CATCHER_NO_UPDATE, CATCHER_OK, catcher_create,
    catcher_destroy, catcher_finish, catcher_last_error, catcher_push_audio, catcher_segments,
    catcher_start, catcher_text, catcher_warning,
};

/// MLX evaluates onto a process-global Metal command buffer that is not safe
/// for concurrent submission; two full-pipeline tests running on separate
/// threads abort with "A command encoder is already encoding to this command
/// buffer". Each MLX-driving test holds this lock so they run serially.
static MLX_PIPELINE: Mutex<()> = Mutex::new(());

fn serialize_mlx() -> MutexGuard<'static, ()> {
    MLX_PIPELINE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn null_arguments_report_errors_without_unwinding() {
    unsafe {
        assert_eq!(catcher_start(ptr::null_mut()), CATCHER_INVALID_ARGUMENT);
        assert_eq!(
            catcher_push_audio(ptr::null_mut(), ptr::null(), 0),
            CATCHER_INVALID_ARGUMENT
        );
        assert_eq!(catcher_finish(ptr::null_mut()), CATCHER_INVALID_ARGUMENT);
        assert!(catcher_text(ptr::null()).is_null());
        assert!(!catcher_last_error(ptr::null()).is_null());
        catcher_destroy(ptr::null_mut());
    }
}

#[test]
fn null_handle_segments_and_warning_are_safe() {
    // Follow catcher_text's NULL-handle convention exactly (lib.rs:171-ish):
    // a null handle returns a null pointer and records a thread-local error,
    // rather than crashing.
    unsafe {
        assert!(catcher_segments(ptr::null()).is_null());
        assert!(!catcher_last_error(ptr::null()).is_null());
        assert!(catcher_warning(ptr::null()).is_null());
        assert!(!catcher_last_error(ptr::null()).is_null());
    }
}

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT"]
fn valid_handle_can_restart_without_reloading() {
    let _guard = serialize_mlx();
    let model = CString::new(std::env::var("NEMOTRON_MLX_ARTIFACT").unwrap()).unwrap();
    let language = CString::new("en-US").unwrap();
    let handle = unsafe { catcher_create(model.as_ptr(), ptr::null(), language.as_ptr(), 3) };
    assert!(!handle.is_null());

    unsafe {
        assert_eq!(catcher_start(handle), CATCHER_OK);
        assert_eq!(
            catcher_push_audio(handle, ptr::null(), 0),
            CATCHER_NO_UPDATE
        );
        assert_eq!(catcher_finish(handle), CATCHER_NO_UPDATE);
        assert_eq!(
            catcher_push_audio(handle, ptr::null(), 0),
            CATCHER_INVALID_STATE
        );
        assert_eq!(CStr::from_ptr(catcher_text(handle)).to_bytes(), b"");

        assert_eq!(catcher_start(handle), CATCHER_OK);
        assert_eq!(CStr::from_ptr(catcher_text(handle)).to_bytes(), b"");
        catcher_destroy(handle);
    }
}

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT"]
fn c_abi_transcribes_reference_wav_exactly() {
    let _guard = serialize_mlx();
    let model = CString::new(std::env::var("NEMOTRON_MLX_ARTIFACT").unwrap()).unwrap();
    let language = CString::new("en-US").unwrap();
    let handle = unsafe { catcher_create(model.as_ptr(), ptr::null(), language.as_ptr(), 3) };
    assert!(!handle.is_null());
    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hello-streaming.wav"
    ))
    .unwrap();
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect::<Vec<_>>();

    unsafe {
        assert_eq!(catcher_start(handle), CATCHER_OK);
        let sizes = [127, 1_024, 333, 2_048];
        let mut offset = 0;
        let mut block = 0;
        while offset < samples.len() {
            let end = (offset + sizes[block % sizes.len()]).min(samples.len());
            let status = catcher_push_audio(handle, samples[offset..end].as_ptr(), end - offset);
            assert!(status == CATCHER_OK || status == CATCHER_NO_UPDATE);
            offset = end;
            block += 1;
        }
        let status = catcher_finish(handle);
        assert!(status == CATCHER_OK || status == CATCHER_NO_UPDATE);
        assert_eq!(
            CStr::from_ptr(catcher_text(handle)).to_str().unwrap(),
            "Hello, this is a streaming speech recognition test"
        );
        catcher_destroy(handle);
    }
}

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT"]
fn ascii_transcription_with_null_diar_keeps_v1_behavior() {
    let _guard = serialize_mlx();
    let model = CString::new(std::env::var("NEMOTRON_MLX_ARTIFACT").unwrap()).unwrap();
    let language = CString::new("en-US").unwrap();
    let handle = unsafe { catcher_create(model.as_ptr(), ptr::null(), language.as_ptr(), 3) };
    assert!(!handle.is_null());
    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hello-streaming.wav"
    ))
    .unwrap();
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect::<Vec<_>>();

    unsafe {
        assert_eq!(catcher_start(handle), CATCHER_OK);
        let sizes = [127, 1_024, 333, 2_048];
        let mut offset = 0;
        let mut block = 0;
        while offset < samples.len() {
            let end = (offset + sizes[block % sizes.len()]).min(samples.len());
            let status = catcher_push_audio(handle, samples[offset..end].as_ptr(), end - offset);
            assert!(status == CATCHER_OK || status == CATCHER_NO_UPDATE);
            // With no diarization model loaded, there is never anything to
            // warn about and segments always read as the never-diarized "[]".
            assert!(catcher_warning(handle).is_null());
            assert_eq!(
                CStr::from_ptr(catcher_segments(handle)).to_str().unwrap(),
                "[]"
            );
            offset = end;
            block += 1;
        }
        let status = catcher_finish(handle);
        assert!(status == CATCHER_OK || status == CATCHER_NO_UPDATE);
        assert_eq!(
            CStr::from_ptr(catcher_text(handle)).to_str().unwrap(),
            "Hello, this is a streaming speech recognition test"
        );
        assert_eq!(
            CStr::from_ptr(catcher_segments(handle)).to_str().unwrap(),
            "[]"
        );
        assert!(catcher_warning(handle).is_null());
        catcher_destroy(handle);
    }
}

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT and SORTFORMER_MLX_ARTIFACT"]
fn dual_model_create_produces_decodable_segments_json() {
    let _guard = serialize_mlx();
    let asr_model = CString::new(std::env::var("NEMOTRON_MLX_ARTIFACT").unwrap()).unwrap();
    let diar_model = CString::new(std::env::var("SORTFORMER_MLX_ARTIFACT").unwrap()).unwrap();
    let language = CString::new("auto").unwrap();
    let handle = unsafe {
        catcher_create(
            asr_model.as_ptr(),
            diar_model.as_ptr(),
            language.as_ptr(),
            3,
        )
    };
    assert!(!handle.is_null());

    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/conversation.wav"
    ))
    .unwrap();
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect::<Vec<_>>();

    unsafe {
        assert_eq!(catcher_start(handle), CATCHER_OK);
        for piece in samples.chunks(1600) {
            let status = catcher_push_audio(handle, piece.as_ptr(), piece.len());
            assert!(
                status == CATCHER_OK || status == CATCHER_NO_UPDATE,
                "unexpected push status {status}"
            );
        }
        let status = catcher_finish(handle);
        assert!(
            status == CATCHER_OK || status == CATCHER_NO_UPDATE,
            "unexpected finish status {status}"
        );

        let segments_json = CStr::from_ptr(catcher_segments(handle))
            .to_str()
            .unwrap()
            .to_string();
        let segments: Vec<serde_json::Value> = serde_json::from_str(&segments_json).unwrap();
        assert!(!segments.is_empty(), "expected at least one segment");

        let mut prev_start_ms: Option<u64> = None;
        for segment in &segments {
            let object = segment.as_object().expect("segment must be a JSON object");
            assert!(object.contains_key("speaker"));
            let start_ms = object
                .get("start_ms")
                .and_then(serde_json::Value::as_u64)
                .expect("start_ms must be present");
            assert!(object.contains_key("end_ms"));
            assert!(object.contains_key("text"));
            assert_eq!(
                object.get("final").and_then(serde_json::Value::as_bool),
                Some(true),
                "every segment must be final after catcher_finish"
            );
            if let Some(prev) = prev_start_ms {
                assert!(
                    start_ms >= prev,
                    "segments must be sorted by start_ms ({start_ms} < {prev})"
                );
            }
            prev_start_ms = Some(start_ms);
        }

        catcher_destroy(handle);
    }
}
