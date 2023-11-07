use ffi_support::{
    call_with_result, define_string_destructor, rust_string_to_c, ErrorCode, ExternError, FfiStr,
};
use libtashkeel_base::{create_inference_engine, do_tashkeel, DynamicInferenceEngine, LibtashkeelError};
use once_cell::sync::OnceCell;
use std::ffi::c_char;
use std::path::PathBuf;
use std::sync::Once;

static INFERENCE_ENGINE: OnceCell<DynamicInferenceEngine> = OnceCell::new();
static INIT_LIBTASHKEEL: Once = Once::new();


#[allow(non_snake_case)]
mod ErrorCodes {
    pub const INPUT_TOO_LONG: i32 = 1;
    pub const INFERENCE_ERROR: i32 = 2;
    pub const MODEL_LOAD_ERROR: i32 = 3;
    pub const UNKNOWN_ERROR: i32 = 99;
}


#[derive(Debug)]
struct LibtashkeelFFIError(i32, String);

impl From<LibtashkeelError> for LibtashkeelFFIError {
    fn from(other: LibtashkeelError) -> Self {
        let (code, message) = match other {
            LibtashkeelError::InputTooLong(max_len) => {
                (ErrorCodes::INPUT_TOO_LONG, format!("Input too long. Max length {}", max_len))
            }
            LibtashkeelError::InferenceError(msg) => (ErrorCodes::INFERENCE_ERROR, msg),
            LibtashkeelError::ModelLoadError(e) => (ErrorCodes::MODEL_LOAD_ERROR, e.to_string()),
        };
        Self(code, message)
    }
}

impl From<LibtashkeelFFIError> for ExternError {
    fn from(other: LibtashkeelFFIError) -> Self {
        let err_code = ErrorCode::new(other.0);
        ExternError::new_error(err_code, other.1)
    }
}

type LibtashkeelFFIResult<T> = Result<T, LibtashkeelFFIError>;

define_string_destructor!(libtashkeel_free_string);

/// # Safety
/// The `taskeen_threshold_ptr` should be properly alighned as `c_float`
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libtashkeelTashkeel(
    text_ptr: FfiStr,
    taskeen_threshold: *const libc::c_float,
    out_error: &mut ExternError,
) -> *mut c_char {
    let text = text_ptr.into_string();
    let taskeen_threshold = unsafe {
        let retval = taskeen_threshold.as_ref().copied();
        libc::free(taskeen_threshold as *mut libc::c_void);
        retval
    };
    call_with_result(out_error, move || {
        INIT_LIBTASHKEEL.call_once(|| {
            do_init_library(None).unwrap();
        });
        let diacritized_text =
            ffi_do_tashkeel(INFERENCE_ENGINE.get().unwrap(), &text, taskeen_threshold)?;
        let retval = rust_string_to_c(diacritized_text);
        Ok::<*mut c_char, LibtashkeelFFIError>(retval)
    })
}

#[no_mangle]
#[allow(non_snake_case)]
pub extern "C" fn libtashkeel_init(model_path_ptr: FfiStr, out_error: &mut ExternError) {
    let model_path = model_path_ptr.into_opt_string().map(PathBuf::from);
    call_with_result(out_error, move || do_init_library(model_path))
}

fn ffi_do_tashkeel(
    model: &DynamicInferenceEngine,
    text: &str,
    taskeen_threshold: Option<f32>,
) -> LibtashkeelFFIResult<String> {
    Ok(do_tashkeel(model, text, taskeen_threshold)?)
}

fn do_init_library(model_path: Option<PathBuf>) -> LibtashkeelFFIResult<()> {
    INIT_LIBTASHKEEL.call_once(|| ());
    let engine = create_inference_engine(model_path)?;
    if INFERENCE_ENGINE
        .set(engine)
        .is_err()
    {
        Err(LibtashkeelFFIError(
            ErrorCodes::UNKNOWN_ERROR,
            "Unexpected error. Failed to init global inference_engine instance with `tract`."
                .to_string(),
        ))
    } else {
        Ok(())
    }
}