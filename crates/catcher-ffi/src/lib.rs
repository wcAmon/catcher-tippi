use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;
use std::slice;

use nemotron_mlx::{
    fusion::{Fusion, FusionConfig},
    model::{StreamingTranscriber, TimedToken},
    opencc,
    tokenizer::Tokenizer,
    weights::Artifact,
};
use sortformer_mlx::stream::StreamingDiarizer;

pub const CATCHER_OK: i32 = 0;
pub const CATCHER_NO_UPDATE: i32 = 1;
pub const CATCHER_INVALID_ARGUMENT: i32 = -1;
pub const CATCHER_INVALID_STATE: i32 = -2;
pub const CATCHER_RUNTIME_ERROR: i32 = -3;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(empty_c_string());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionState {
    Idle,
    Started,
    Finished,
}

#[repr(C)]
pub struct CatcherHandle {
    transcriber: StreamingTranscriber,
    tokenizer: Tokenizer,
    /// `Some` only while a diarizer was requested at `catcher_create` *and*
    /// has not yet hit a runtime error. Degrades to `None` on a diarization
    /// runtime failure without affecting transcription.
    diarizer: Option<StreamingDiarizer>,
    /// Whether `catcher_create` was given a non-null `diar_model_path`. Kept
    /// separately from `diarizer` because `diarizer` can degrade to `None`
    /// mid-session while `catcher_segments` must keep reporting real (if
    /// stale) segments rather than snapping back to the never-diarized `[]`.
    diarization_requested: bool,
    fusion: Fusion,
    timed_tokens: Vec<TimedToken>,
    text: CString,
    segments_json: CString,
    warning: Option<CString>,
    last_error: CString,
    state: SessionState,
}

/// Loads a Catcher ASR model (and, optionally, a Sortformer diarization
/// model) and creates an idle transcription handle.
///
/// # Safety
///
/// `asr_model_path` and `language` must point to valid NUL-terminated UTF-8
/// strings. `diar_model_path` must be null or point to a valid
/// NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_create(
    asr_model_path: *const c_char,
    diar_model_path: *const c_char,
    language: *const c_char,
    lookahead: u32,
) -> *mut CatcherHandle {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let model_path = unsafe { required_string(asr_model_path, "asr_model_path")? };
        let language = unsafe { required_string(language, "language")? };
        let diar_model_path = unsafe { optional_string(diar_model_path, "diar_model_path")? };
        let artifact = Artifact::load(&model_path).map_err(|error| error.to_string())?;
        let transcriber = StreamingTranscriber::new(&artifact, &language, lookahead as usize)
            .map_err(|error| error.to_string())?;
        let tokenizer =
            Tokenizer::from_json(Path::new(&model_path).join("tokenizer.json"), 0, 13_087)
                .map_err(|error| error.to_string())?;
        let diarization_requested = diar_model_path.is_some();
        let diarizer = match diar_model_path {
            Some(diar_model_path) => Some(
                StreamingDiarizer::from_artifact_dir(&diar_model_path)
                    .map_err(|error| error.to_string())?,
            ),
            None => None,
        };
        Ok::<_, String>(CatcherHandle {
            transcriber,
            tokenizer,
            diarizer,
            diarization_requested,
            fusion: Fusion::new(FusionConfig::default()),
            timed_tokens: Vec::new(),
            text: empty_c_string(),
            segments_json: safe_c_string("[]"),
            warning: None,
            last_error: empty_c_string(),
            state: SessionState::Idle,
        })
    }));

    match result {
        Ok(Ok(handle)) => Box::into_raw(Box::new(handle)),
        Ok(Err(error)) => {
            set_global_error(&error);
            ptr::null_mut()
        }
        Err(payload) => {
            set_global_error(&panic_message(payload));
            ptr::null_mut()
        }
    }
}

/// Clears caches and text for a new utterance.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `catcher_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_start(handle: *mut CatcherHandle) -> i32 {
    unsafe {
        with_handle_mut(handle, |handle| {
            handle
                .transcriber
                .reset()
                .map_err(|error| (CATCHER_RUNTIME_ERROR, error.to_string()))?;
            if let Some(diarizer) = handle.diarizer.as_mut() {
                diarizer.reset();
            }
            handle.fusion.reset();
            handle.timed_tokens.clear();
            handle.text = empty_c_string();
            handle.warning = None;
            handle.segments_json = safe_c_string("[]");
            handle.state = SessionState::Started;
            Ok(CATCHER_OK)
        })
    }
}

/// Pushes arbitrary mono Float32 16 kHz samples into the active utterance.
///
/// Returns `CATCHER_OK` when new ASR tokens were decoded from this call *or*
/// `catcher_segments` changed as a result of it (diarization can re-attribute
/// or finalize a tentative trailing segment on diarization-only audio, with
/// no new ASR tokens at all); returns `CATCHER_NO_UPDATE` only when neither
/// happened. Callers must not skip re-reading `catcher_segments` on
/// `CATCHER_NO_UPDATE` under the assumption that "no new tokens" implies "no
/// new segments" — that assumption held for the ASR-only v1 API but not once
/// a diarization model is attached.
///
/// # Safety
///
/// `handle` must be live. When `count` is non-zero, `samples` must reference at
/// least `count` initialized floats for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_push_audio(
    handle: *mut CatcherHandle,
    samples: *const f32,
    count: usize,
) -> i32 {
    if count > 0 && samples.is_null() {
        set_global_error("samples is null while count is non-zero");
        return CATCHER_INVALID_ARGUMENT;
    }
    let samples = if count == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(samples, count) }
    };
    unsafe {
        with_handle_mut(handle, |handle| {
            if handle.state != SessionState::Started {
                return Err((
                    CATCHER_INVALID_STATE,
                    "catcher session is not recording".to_string(),
                ));
            }
            let tokens = handle
                .transcriber
                .push_samples(samples)
                .map_err(|error| (CATCHER_RUNTIME_ERROR, error.to_string()))?;
            let has_new_tokens = !tokens.is_empty();
            if has_new_tokens {
                handle.fusion.push_tokens(&tokens);
                handle.timed_tokens.extend(tokens);
            }
            push_diar_samples(handle, samples);
            let segments_changed = rebuild_strings_and_report_segment_change(handle)?;
            Ok(if has_new_tokens || segments_changed {
                CATCHER_OK
            } else {
                CATCHER_NO_UPDATE
            })
        })
    }
}

/// Flushes the final partial audio window and locks the current utterance.
///
/// Same `CATCHER_OK`/`CATCHER_NO_UPDATE` semantics as `catcher_push_audio`:
/// `catcher_finish` always forces every trailing tentative segment final,
/// which almost always changes `catcher_segments` even with no new ASR
/// tokens, so `CATCHER_OK` is the common case for a diarization-enabled
/// handle.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `catcher_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_finish(handle: *mut CatcherHandle) -> i32 {
    unsafe {
        with_handle_mut(handle, |handle| {
            if handle.state != SessionState::Started {
                return Err((
                    CATCHER_INVALID_STATE,
                    "catcher session is not recording".to_string(),
                ));
            }
            let tokens = handle
                .transcriber
                .finish()
                .map_err(|error| (CATCHER_RUNTIME_ERROR, error.to_string()))?;
            let has_new_tokens = !tokens.is_empty();
            if has_new_tokens {
                handle.fusion.push_tokens(&tokens);
                handle.timed_tokens.extend(tokens);
            }
            finish_diar(handle);
            handle.fusion.flush();
            handle.state = SessionState::Finished;
            let segments_changed = rebuild_strings_and_report_segment_change(handle)?;
            Ok(if has_new_tokens || segments_changed {
                CATCHER_OK
            } else {
                CATCHER_NO_UPDATE
            })
        })
    }
}

/// Returns the current UTF-8 transcript owned by `handle`.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `catcher_create`. The
/// returned pointer is invalidated by the next mutating call or destroy.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_text(handle: *const CatcherHandle) -> *const c_char {
    if handle.is_null() {
        set_global_error("catcher handle is null");
        return ptr::null();
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe { (&*handle).text.as_ptr() })) {
        Ok(pointer) => pointer,
        Err(payload) => {
            set_global_error(&panic_message(payload));
            ptr::null()
        }
    }
}

/// Returns the current speaker segments as a UTF-8 JSON array owned by
/// `handle` (`[]` when no diarization model was supplied to
/// `catcher_create`).
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `catcher_create`. The
/// returned pointer is invalidated by the next mutating call or destroy,
/// exactly like `catcher_text`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_segments(handle: *const CatcherHandle) -> *const c_char {
    if handle.is_null() {
        set_global_error("catcher handle is null");
        return ptr::null();
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        (&*handle).segments_json.as_ptr()
    })) {
        Ok(pointer) => pointer,
        Err(payload) => {
            set_global_error(&panic_message(payload));
            ptr::null()
        }
    }
}

/// Returns the current non-fatal diarization warning owned by `handle`, or
/// null when there is none (e.g. no diarization model was supplied, or
/// nothing has gone wrong yet).
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `catcher_create`. The
/// returned pointer is invalidated by the next mutating call or destroy,
/// exactly like `catcher_text`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_warning(handle: *const CatcherHandle) -> *const c_char {
    if handle.is_null() {
        set_global_error("catcher handle is null");
        return ptr::null();
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        (&*handle)
            .warning
            .as_ref()
            .map(|warning| warning.as_ptr())
            .unwrap_or(ptr::null())
    })) {
        Ok(pointer) => pointer,
        Err(payload) => {
            set_global_error(&panic_message(payload));
            ptr::null()
        }
    }
}

/// Returns the last error for a handle, or the current thread when handle is null.
///
/// # Safety
///
/// A non-null `handle` must be a live pointer returned by `catcher_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_last_error(handle: *const CatcherHandle) -> *const c_char {
    if handle.is_null() {
        return LAST_ERROR.with(|error| error.borrow().as_ptr());
    }
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        (&*handle).last_error.as_ptr()
    })) {
        Ok(pointer) => pointer,
        Err(payload) => {
            set_global_error(&panic_message(payload));
            LAST_ERROR.with(|error| error.borrow().as_ptr())
        }
    }
}

/// Releases a Catcher handle. A null pointer is accepted.
///
/// # Safety
///
/// A non-null `handle` must be a live pointer returned by `catcher_create` and
/// must not be destroyed more than once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_destroy(handle: *mut CatcherHandle) {
    if handle.is_null() {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(handle));
    }));
    if let Err(payload) = result {
        set_global_error(&panic_message(payload));
    }
}

/// Pushes `samples` into the diarizer, if one is still attached. A runtime
/// diarization error demotes `diarizer` to `None` and records a warning;
/// transcription is unaffected and the caller should treat this as
/// non-fatal.
fn push_diar_samples(handle: &mut CatcherHandle, samples: &[f32]) {
    let Some(diarizer) = handle.diarizer.as_mut() else {
        return;
    };
    match diarizer.push_samples(samples) {
        Ok(frames) => {
            if !frames.is_empty() {
                handle.fusion.push_diar_frames(&frames);
            }
        }
        Err(error) => {
            handle.warning = Some(diarizer_disabled_warning(&error));
            handle.diarizer = None;
        }
    }
}

/// Flushes the diarizer's trailing chunks, if one is still attached. Same
/// non-fatal degrade-and-warn behavior as [`push_diar_samples`].
fn finish_diar(handle: &mut CatcherHandle) {
    let Some(diarizer) = handle.diarizer.as_mut() else {
        return;
    };
    match diarizer.finish() {
        Ok(frames) => {
            if !frames.is_empty() {
                handle.fusion.push_diar_frames(&frames);
            }
        }
        Err(error) => {
            handle.warning = Some(diarizer_disabled_warning(&error));
            handle.diarizer = None;
        }
    }
}

/// Rebuilds `handle.text` (s2twp-converted full transcript) and
/// `handle.segments_json` from the accumulated tokens/diarization frames, and
/// reports whether `segments_json` came out different from what it was
/// before this call. Diarization-only audio (no new ASR tokens) can still
/// re-attribute or finalize a tentative trailing segment, so callers must not
/// assume "no new tokens" implies "no update to report" — see
/// `CATCHER_NO_UPDATE`'s doc comment in the header.
fn rebuild_strings_and_report_segment_change(
    handle: &mut CatcherHandle,
) -> Result<bool, (i32, String)> {
    let previous_segments_json = handle.segments_json.as_bytes().to_vec();

    let ids: Vec<u32> = handle.timed_tokens.iter().map(|token| token.id).collect();
    let decoded = handle
        .tokenizer
        .decode(&ids, true)
        .map_err(|error| (CATCHER_RUNTIME_ERROR, error.to_string()))?;
    handle.text = safe_c_string(&opencc::to_traditional(&decoded));

    let segments_json = if handle.diarization_requested {
        let tokenizer = &handle.tokenizer;
        let segments = handle.fusion.segments(|ids| {
            tokenizer
                .decode(ids, true)
                .map(|decoded| opencc::to_traditional(&decoded))
                .unwrap_or_default()
        });
        serde_json::to_string(&segments).expect("SpeakerSegment serialization cannot fail")
    } else {
        "[]".to_string()
    };
    handle.segments_json = safe_c_string(&segments_json);

    Ok(handle.segments_json.as_bytes() != previous_segments_json.as_slice())
}

unsafe fn with_handle_mut(
    handle: *mut CatcherHandle,
    operation: impl FnOnce(&mut CatcherHandle) -> Result<i32, (i32, String)>,
) -> i32 {
    if handle.is_null() {
        set_global_error("catcher handle is null");
        return CATCHER_INVALID_ARGUMENT;
    }
    match catch_unwind(AssertUnwindSafe(|| operation(unsafe { &mut *handle }))) {
        Ok(Ok(status)) => status,
        Ok(Err((status, error))) => {
            unsafe { &mut *handle }.last_error = safe_c_string(&error);
            status
        }
        Err(payload) => {
            unsafe { &mut *handle }.last_error = safe_c_string(&panic_message(payload));
            CATCHER_RUNTIME_ERROR
        }
    }
}

unsafe fn required_string(pointer: *const c_char, name: &str) -> Result<String, String> {
    if pointer.is_null() {
        return Err(format!("{name} is null"));
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|_| format!("{name} is not valid UTF-8"))
}

unsafe fn optional_string(pointer: *const c_char, name: &str) -> Result<Option<String>, String> {
    if pointer.is_null() {
        return Ok(None);
    }
    unsafe { required_string(pointer, name) }.map(Some)
}

fn empty_c_string() -> CString {
    CString::new(Vec::new()).expect("empty C string is valid")
}

fn safe_c_string(message: &str) -> CString {
    CString::new(message.replace('\0', "�")).expect("NUL bytes were replaced")
}

/// Formats the warning stored when a diarizer degrades to `None` after a
/// runtime error. Kept as one place so `catcher_warning`'s wording (see the
/// header doc comment) stays in sync with what actually gets stored.
fn diarizer_disabled_warning(error: &sortformer_mlx::model::ModelError) -> CString {
    safe_c_string(&format!(
        "diarization disabled after a runtime error: {error}"
    ))
}

fn set_global_error(message: &str) {
    LAST_ERROR.with(|error| *error.borrow_mut() = safe_c_string(message));
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown Rust panic in Catcher".to_string()
    }
}
