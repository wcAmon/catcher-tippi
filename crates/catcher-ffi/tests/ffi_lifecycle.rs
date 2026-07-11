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
