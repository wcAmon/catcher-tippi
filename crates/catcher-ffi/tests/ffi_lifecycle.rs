use std::ffi::{CStr, CString};
use std::ptr;

use catcher_ffi::{
    CATCHER_INVALID_ARGUMENT, CATCHER_INVALID_STATE, CATCHER_NO_UPDATE, CATCHER_OK, catcher_create,
    catcher_destroy, catcher_finish, catcher_last_error, catcher_push_audio, catcher_start,
    catcher_text,
};

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
#[ignore = "requires NEMOTRON_MLX_ARTIFACT"]
fn valid_handle_can_restart_without_reloading() {
    let model = CString::new(std::env::var("NEMOTRON_MLX_ARTIFACT").unwrap()).unwrap();
    let language = CString::new("en-US").unwrap();
    let handle = unsafe { catcher_create(model.as_ptr(), language.as_ptr(), 3) };
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
    let model = CString::new(std::env::var("NEMOTRON_MLX_ARTIFACT").unwrap()).unwrap();
    let language = CString::new("en-US").unwrap();
    let handle = unsafe { catcher_create(model.as_ptr(), language.as_ptr(), 3) };
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
