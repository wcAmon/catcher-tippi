use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;
use std::slice;

use nemotron_mlx::{model::StreamingTranscriber, tokenizer::Tokenizer, weights::Artifact};

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
    tokens: Vec<u32>,
    text: CString,
    last_error: CString,
    state: SessionState,
}

/// Loads a Catcher model and creates an idle transcription handle.
///
/// # Safety
///
/// `model_path` and `language` must point to valid NUL-terminated UTF-8 strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_create(
    model_path: *const c_char,
    language: *const c_char,
    lookahead: u32,
) -> *mut CatcherHandle {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let model_path = unsafe { required_string(model_path, "model_path")? };
        let language = unsafe { required_string(language, "language")? };
        let artifact = Artifact::load(&model_path).map_err(|error| error.to_string())?;
        let transcriber = StreamingTranscriber::new(&artifact, &language, lookahead as usize)
            .map_err(|error| error.to_string())?;
        let tokenizer =
            Tokenizer::from_json(Path::new(&model_path).join("tokenizer.json"), 0, 13_087)
                .map_err(|error| error.to_string())?;
        Ok::<_, String>(CatcherHandle {
            transcriber,
            tokenizer,
            tokens: Vec::new(),
            text: empty_c_string(),
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
            handle.tokens.clear();
            handle.text = empty_c_string();
            handle.state = SessionState::Started;
            Ok(CATCHER_OK)
        })
    }
}

/// Pushes arbitrary mono Float32 16 kHz samples into the active utterance.
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
            update_text(handle, tokens)
        })
    }
}

/// Flushes the final partial audio window and locks the current utterance.
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
            handle.state = SessionState::Finished;
            update_text(handle, tokens)
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

fn update_text(handle: &mut CatcherHandle, tokens: Vec<u32>) -> Result<i32, (i32, String)> {
    if tokens.is_empty() {
        return Ok(CATCHER_NO_UPDATE);
    }
    handle.tokens.extend(tokens);
    let text = handle
        .tokenizer
        .decode(&handle.tokens, true)
        .map_err(|error| (CATCHER_RUNTIME_ERROR, error.to_string()))?;
    handle.text = safe_c_string(&text);
    Ok(CATCHER_OK)
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

fn empty_c_string() -> CString {
    CString::new(Vec::new()).expect("empty C string is valid")
}

fn safe_c_string(message: &str) -> CString {
    CString::new(message.replace('\0', "�")).expect("NUL bytes were replaced")
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
