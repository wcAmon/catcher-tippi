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

/// Loads the shared dual-model test fixture as mono Float32 16 kHz samples.
fn conversation_samples() -> Vec<f32> {
    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/conversation.wav"
    ))
    .unwrap();
    reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect::<Vec<_>>()
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

    let samples = conversation_samples();

    unsafe {
        assert_eq!(catcher_start(handle), CATCHER_OK);
        for piece in samples.chunks(1600) {
            // Capture `catcher_segments` before the push so a `CATCHER_NO_UPDATE`
            // result can be checked against the invariant it is supposed to
            // guarantee: segments did NOT change. This is the regression test
            // for the NO_UPDATE-vs-segments bug, where diar-only re-attribution
            // could silently change segments_json while reporting NO_UPDATE
            // (because only new-token arrival was checked).
            let segments_before = CStr::from_ptr(catcher_segments(handle))
                .to_str()
                .unwrap()
                .to_string();
            let status = catcher_push_audio(handle, piece.as_ptr(), piece.len());
            assert!(
                status == CATCHER_OK || status == CATCHER_NO_UPDATE,
                "unexpected push status {status}"
            );
            let segments_after = CStr::from_ptr(catcher_segments(handle))
                .to_str()
                .unwrap()
                .to_string();
            if status == CATCHER_NO_UPDATE {
                assert_eq!(
                    segments_before, segments_after,
                    "CATCHER_NO_UPDATE must imply catcher_segments did not change"
                );
            }
        }
        let segments_before_finish = CStr::from_ptr(catcher_segments(handle))
            .to_str()
            .unwrap()
            .to_string();
        let status = catcher_finish(handle);
        assert!(
            status == CATCHER_OK || status == CATCHER_NO_UPDATE,
            "unexpected finish status {status}"
        );
        if status == CATCHER_NO_UPDATE {
            let segments_after_finish = CStr::from_ptr(catcher_segments(handle))
                .to_str()
                .unwrap()
                .to_string();
            assert_eq!(
                segments_before_finish, segments_after_finish,
                "CATCHER_NO_UPDATE must imply catcher_segments did not change"
            );
        }

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

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT and SORTFORMER_MLX_ARTIFACT"]
fn degraded_diarizer_is_rebuilt_on_next_start() {
    let _guard = serialize_mlx();
    let model = CString::new(std::env::var("NEMOTRON_MLX_ARTIFACT").unwrap()).unwrap();
    let diar_model = CString::new(std::env::var("SORTFORMER_MLX_ARTIFACT").unwrap()).unwrap();
    let language = CString::new("auto").unwrap();
    let handle =
        unsafe { catcher_create(model.as_ptr(), diar_model.as_ptr(), language.as_ptr(), 3) };
    assert!(!handle.is_null());
    let samples = conversation_samples();

    unsafe {
        assert_eq!(catcher_start(handle), CATCHER_OK);
        catcher_push_audio(handle, samples.as_ptr(), 16_000);
        assert!(catcher_warning(handle).is_null());

        // 注入執行期降級:warning 出現、後續 push 仍可運作(純 ASR)。
        catcher_ffi::test_degrade_diarizer(handle);
        assert!(!catcher_warning(handle).is_null());
        let status = catcher_push_audio(handle, samples[16_000..].as_ptr(), 16_000);
        assert!(status == CATCHER_OK || status == CATCHER_NO_UPDATE);
        assert!(!catcher_warning(handle).is_null());
        catcher_finish(handle);

        // 下一次 start 就地重建:warning 清空、diarization 恢復產出 segments。
        assert_eq!(catcher_start(handle), CATCHER_OK);
        assert!(catcher_warning(handle).is_null());
        let mut offset = 0usize;
        let chunk = 16_000;
        let mut recovered = false;
        while offset + chunk <= samples.len() {
            catcher_push_audio(handle, samples[offset..].as_ptr(), chunk);
            offset += chunk;
            let segments = CStr::from_ptr(catcher_segments(handle));
            if segments.to_bytes() != b"[]" {
                recovered = true;
                break;
            }
        }
        assert!(recovered, "diarization produced no segments after rebuild");
        catcher_destroy(handle);
    }
}

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT and SORTFORMER_MLX_ARTIFACT"]
fn failed_rebuild_keeps_warning_and_start_succeeds() {
    let _guard = serialize_mlx();
    // 把 diar artifact 複製到暫存目錄,degrade 後刪除,迫使重建失敗。
    let source = std::path::PathBuf::from(std::env::var("SORTFORMER_MLX_ARTIFACT").unwrap());
    let staging = std::env::temp_dir().join(format!("catcher-ffi-rebuild-{}", std::process::id()));
    std::fs::create_dir_all(&staging).unwrap();
    for entry in std::fs::read_dir(&source).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), staging.join(entry.file_name())).unwrap();
    }

    let model = CString::new(std::env::var("NEMOTRON_MLX_ARTIFACT").unwrap()).unwrap();
    let diar_model = CString::new(staging.to_str().unwrap()).unwrap();
    let language = CString::new("auto").unwrap();
    let handle =
        unsafe { catcher_create(model.as_ptr(), diar_model.as_ptr(), language.as_ptr(), 3) };
    assert!(!handle.is_null());

    unsafe {
        assert_eq!(catcher_start(handle), CATCHER_OK);
        catcher_ffi::test_degrade_diarizer(handle);
        std::fs::remove_dir_all(&staging).unwrap();

        // 重建失敗必須非致命:start 回 OK、warning 保留、純 ASR 繼續可用。
        assert_eq!(catcher_start(handle), CATCHER_OK);
        assert!(!catcher_warning(handle).is_null());
        let samples = conversation_samples();
        let status = catcher_push_audio(handle, samples.as_ptr(), 16_000);
        assert!(status == CATCHER_OK || status == CATCHER_NO_UPDATE);
        catcher_destroy(handle);
    }
}
