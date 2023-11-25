use ffi_support::{call_with_result, define_string_destructor, ErrorCode, ExternError, FfiStr};
use once_cell::sync::OnceCell;
use sonata_core::{AudioSamples, SonataError, SonataModel, SonataResult};
use sonata_synth::{AudioOutputConfig, SonataSpeechSynthesizer, SYNTHESIS_THREAD_POOL};
use std::any::Any;
use std::ops::Deref;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::Arc;

pub type SpeechSynthesisCallback = extern "C" fn(LibsonataBuffer) -> bool;
static ORT_ENVIRONMENT: OnceCell<Arc<ort::Environment>> = OnceCell::new();

define_string_destructor!(_internal_libsonataFreeString);
ffi_support::implement_into_ffi_by_pointer!(SonataVoice);
ffi_support::define_box_destructor!(SonataVoice, _internal_libsonataUnloadSonataVoice);
ffi_support::implement_into_ffi_by_pointer!(PiperSynthConfig);
ffi_support::define_box_destructor!(PiperSynthConfig, _internal_libsonataFreePiperSynthConfig);

pub mod error_codes {
    pub const FAILED_TO_LOAD_RESOURCE: i32 = 17;
    pub const PHONEMIZATION_ERROR: i32 = 18;
    pub const OPERATION_ERROR: i32 = 19;
    pub const INVALID_UTF8_SEQUENCE: i32 = 20;
    pub const UNKNOWN_ERROR: i32 = 21;
}

pub struct SonataVoice(AssertUnwindSafe<Arc<SonataSpeechSynthesizer>>);

impl From<SonataSpeechSynthesizer> for SonataVoice {
    fn from(other: SonataSpeechSynthesizer) -> Self {
        Self(AssertUnwindSafe(Arc::new(other)))
    }
}

impl Deref for SonataVoice {
    type Target = SonataSpeechSynthesizer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> AsRef<T> for SonataVoice
where
    T: ?Sized,
    <SonataVoice as Deref>::Target: AsRef<T>,
{
    fn as_ref(&self) -> &T {
        self.deref().as_ref()
    }
}

#[derive(Debug)]
pub struct SonataFFIError(i32, String);

impl SonataFFIError {
    fn invalid_utf8() -> Self {
        Self(
            error_codes::INVALID_UTF8_SEQUENCE,
            "Invalid utf-8 input.".to_string(),
        )
    }
}

impl From<SonataError> for SonataFFIError {
    fn from(other: SonataError) -> Self {
        let (code, message) = match other {
            SonataError::FailedToLoadResource(msg) => (error_codes::FAILED_TO_LOAD_RESOURCE, msg),
            SonataError::PhonemizationError(msg) => (error_codes::PHONEMIZATION_ERROR, msg),
            SonataError::OperationError(msg) => (error_codes::OPERATION_ERROR, msg),
        };
        Self(code, message)
    }
}

impl From<SonataFFIError> for ExternError {
    fn from(other: SonataFFIError) -> Self {
        let err_code = ErrorCode::new(other.0);
        ExternError::new_error(err_code, other.1)
    }
}

pub type SonataFFIResult<T> = Result<T, SonataFFIError>;

#[derive(Clone)]
#[repr(C)]
pub enum SynthesisMode {
    LAZY = 0,
    PARALLEL = 1,
    REALTIME = 2,
}

#[repr(C)]
pub struct LibsonataBuffer {
    len: i64, // usize causes issues with JNI
    data: *mut u8,
    error_ptr: *mut ExternError,
}

impl From<Vec<u8>> for LibsonataBuffer {
    fn from(other: Vec<u8>) -> Self {
        let mut buf = other.into_boxed_slice();
        let data = buf.as_mut_ptr();
        let len = buf.len();
        std::mem::forget(buf);
        Self {
            len: len as i64,
            data,
            error_ptr: std::ptr::null_mut(),
        }
    }
}

#[repr(C)]
pub struct AudioInfo {
    sample_rate: u32,
    num_channels: u32,
    sample_width: u32,
}

#[derive(Clone)]
#[repr(C)]
pub struct SynthesisParams {
    mode: SynthesisMode,
    rate: u8,
    volume: u8,
    pitch: u8,
    appended_silence_ms: u32,
    callback: SpeechSynthesisCallback,
    nonblocking: bool,
}

impl SynthesisParams {
    fn as_synth_output_config(&self) -> AudioOutputConfig {
        AudioOutputConfig {
            rate: Some(self.rate),
            volume: Some(self.volume),
            pitch: Some(self.pitch),
            appended_silence_ms: Some(self.appended_silence_ms),
        }
    }
}

#[repr(C)]
pub struct PiperSynthConfig {
    speaker: u32,
    length_scale: f32,
    noise_scale: f32,
    noise_w: f32,
}

impl PiperSynthConfig {
    fn as_piper_synth_config(&self) -> sonata_piper::PiperSynthesisConfig {
        sonata_piper::PiperSynthesisConfig {
            speaker: Some(self.speaker.into()),
            noise_scale: self.noise_scale,
            length_scale: self.length_scale,
            noise_w: self.noise_w,
        }
    }
}

/// # Safety
/// Pointer must be non-null and well alighned
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataFreeString(string_ptr: *mut i8) {
    _internal_libsonataFreeString(string_ptr)
}

/// # Safety
/// Pointer must be non-null and well alighned
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataFreePiperSynthConfig(synth_config: *mut PiperSynthConfig) {
    _internal_libsonataFreePiperSynthConfig(synth_config)
}
/// # Safety
/// Pointer must be non-null and well alighned

#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataFreeLibsonataBuffer(buf: LibsonataBuffer) {
    ffi_support::abort_on_panic::with_abort_on_panic(|| {
        if !buf.error_ptr.is_null() {
            drop(Box::from_raw(buf.error_ptr));
        }
        let s = std::slice::from_raw_parts_mut(buf.data, buf.len as usize);
        drop(Box::from_raw(s as *mut [u8]));
    });
}

#[no_mangle]
#[allow(non_snake_case)]
pub extern "C" fn libsonataLoadVoiceFromConfigPath(
    config_path_ptr: FfiStr,
    out_error: &mut ExternError,
) -> *mut SonataVoice {
    call_with_result(out_error, move || _load_piper_voice(config_path_ptr))
}

/// # Safety
/// Pointer must be non-null and well alighned
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataUnloadSonataVoice(voice_ptr: *mut SonataVoice) {
    _internal_libsonataUnloadSonataVoice(voice_ptr)
}

/// # Safety
/// Pointer must be non-null and well alighned
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataGetAudioInfo(
    voice_ptr: *mut SonataVoice,
    audio_info_ptr: *mut AudioInfo,
    out_error: &mut ExternError,
) {
    let voice = voice_ptr.as_ref().unwrap();
    let audio_info_mut = audio_info_ptr.as_mut().unwrap();
    let mut audio_info = AssertUnwindSafe(audio_info_mut);
    call_with_result(out_error, move || {
        voice
            .audio_output_info()
            .map(|a| {
                audio_info.sample_rate = a.sample_rate as u32;
                audio_info.num_channels = a.num_channels as u32;
                audio_info.sample_width = a.sample_width as u32;
            })
            .map_err(SonataFFIError::from)
    })
}

/// # Safety
/// Pointer must be non-null and well alighned
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataGetPiperDefaultSynthConfig(
    voice_ptr: *mut SonataVoice,
    out_error: &mut ExternError,
) -> *mut PiperSynthConfig {
    let voice = voice_ptr.as_ref().unwrap();
    call_with_result(out_error, move || {
        let synth_config = voice
            .get_default_synthesis_config()
            .map_err(SonataFFIError::from)?;
        match synth_config.downcast_ref::<sonata_piper::PiperSynthesisConfig>() {
            Some(config) => Ok(PiperSynthConfig {
                speaker: config.speaker.map(|sid| sid as u32).unwrap_or_default(),
                length_scale: config.length_scale,
                noise_scale: config.noise_scale,
                noise_w: config.noise_w,
            }),
            None => Err(SonataFFIError(
                error_codes::UNKNOWN_ERROR,
                "Cannot retrieve Piper's default synthesis config".to_string(),
            )),
        }
    })
}

/// # Safety
/// Pointer must be non-null and well alighned
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataSetPiperSynthConfig(
    voice_ptr: *mut SonataVoice,
    synth_config: PiperSynthConfig,
    out_error: &mut ExternError,
) {
    let voice = voice_ptr.as_ref().unwrap();
    call_with_result(out_error, move || {
        let piper_synth_config = synth_config.as_piper_synth_config();
        let config = &piper_synth_config as &dyn Any;
        voice
            .set_fallback_synthesis_config(config)
            .map_err(SonataFFIError::from)
    })
}

/// # Safety
/// Pointer must be non-null and well alighned
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataSpeak(
    voice_ptr: *mut SonataVoice,
    text_ptr: FfiStr,
    params: SynthesisParams,
    out_error: &mut ExternError,
) {
    let voice = voice_ptr.as_ref().unwrap();
    let synth = AssertUnwindSafe(Arc::clone(&voice.0));
    call_with_result(out_error, move || _synthesize(synth, text_ptr, params))
}

fn get_ort_environment() -> &'static Arc<ort::Environment> {
    ORT_ENVIRONMENT.get_or_init(|| {
        let execution_providers = [
            #[cfg(target_os = "android")]
            ort::ExecutionProvider::NNAPI(Default::default()),
            #[cfg(target_os = "ios")]
            ort::ExecutionProvider::CoreML(Default::default()),
            ort::ExecutionProvider::CPU(Default::default()),
        ];
        Arc::new(
            ort::Environment::builder()
                .with_name("sonata")
                .with_execution_providers(execution_providers)
                .build()
                .unwrap(),
        )
    })
}

fn _load_piper_voice(config_path_ptr: FfiStr) -> SonataFFIResult<SonataVoice> {
    let config_path = config_path_ptr
        .into_opt_string()
        .ok_or_else(SonataFFIError::invalid_utf8)?;
    let config_path = PathBuf::from(config_path);
    let piper_model = sonata_piper::from_config_path(&config_path, get_ort_environment())?;
    let synth = SonataSpeechSynthesizer::new(piper_model)?;
    Ok(synth.into())
}

fn _synthesize(
    synth: AssertUnwindSafe<Arc<SonataSpeechSynthesizer>>,
    text_ptr: FfiStr,
    params: SynthesisParams,
) -> SonataFFIResult<()> {
    let text = text_ptr
        .into_opt_string()
        .ok_or_else(SonataFFIError::invalid_utf8)?;
    if params.nonblocking {
        SYNTHESIS_THREAD_POOL.spawn(move || {
            let callback = params.callback;
            if let Err(e) = _do_synthesize(synth, text, params) {
                let mut buf: LibsonataBuffer = Vec::<u8>::with_capacity(0).into();
                buf.error_ptr = Box::into_raw(Box::new(e.into()));
                callback(buf);
            }
        });
    } else {
        _do_synthesize(synth, text, params)?;
    }
    Ok(())
}

fn _do_synthesize(
    synth: AssertUnwindSafe<Arc<SonataSpeechSynthesizer>>,
    text: String,
    params: SynthesisParams,
) -> SonataFFIResult<()> {
    let audio_output_config = Some(params.as_synth_output_config());
    match params.mode {
        SynthesisMode::LAZY => {
            let stream = synth
                .synthesize_lazy(text, audio_output_config)?
                .map(|wr| wr.map(|aud| aud.samples));
            iterate_stream(stream, params.callback)
        }
        SynthesisMode::PARALLEL => {
            let stream = synth
                .synthesize_parallel(text, audio_output_config)?
                .map(|wr| wr.map(|aud| aud.samples));
            iterate_stream(stream, params.callback)
        }
        SynthesisMode::REALTIME => {
            let stream = synth.synthesize_streamed(text, audio_output_config, 72, 3)?;
            iterate_stream(stream, params.callback)
        }
    }
}

#[inline(always)]
fn iterate_stream(
    stream: impl Iterator<Item = SonataResult<AudioSamples>> + Send + Sync + 'static,
    callback: SpeechSynthesisCallback,
) -> SonataFFIResult<()> {
    for result in stream {
        match result {
            Ok(audio) => {
                let wav_bytes = audio.as_wave_bytes();
                if !callback(wav_bytes.into()) {
                    return Ok(());
                }
            }
            Err(e) => {
                let mut buf: LibsonataBuffer = Vec::<u8>::with_capacity(0).into();
                let error = SonataFFIError::from(e).into();
                buf.error_ptr = Box::into_raw(Box::new(error));
                callback(buf);
                break;
            }
        };
    }
    Ok(())
}
