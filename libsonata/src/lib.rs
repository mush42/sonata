use ffi_support::{call_with_result, define_string_destructor, ErrorCode, ExternError, FfiStr};
use once_cell::sync::OnceCell;
use sonata_core::{AudioSamples, SonataError, SonataModel, SonataResult};
use sonata_piper::PiperSynthesisConfig;
use sonata_synth::{AudioOutputConfig, SonataSpeechSynthesizer, SYNTHESIS_THREAD_POOL};
use std::any::Any;
use std::ops::Deref;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::channel;


pub type SpeechSynthesisCallback = extern "C" fn(ByteBuffer) -> bool;
static ORT_ENVIRONMENT: OnceCell<Arc<ort::Environment>> = OnceCell::new();

define_string_destructor!(_internal_libsonataFreeString);
ffi_support::implement_into_ffi_by_pointer!(SonataVoice);
ffi_support::define_box_destructor!(SonataVoice, _internal_libsonataUnloadSonataVoice);

pub mod error_codes {
    pub const FAILED_TO_LOAD_RESOURCE: i32 = 17;
    pub const PHONEMIZATION_ERROR: i32 = 18;
    pub const OPERATION_ERROR: i32 = 19;
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

#[repr(C)]
pub enum SynthesisMode {
    LAZY = 0,
    PARALLEL = 1,
    BATCHED = 2,
    REALTIME = 3,
}

#[repr(C)]
pub struct ByteBuffer {
    len: i64, // usize causes issues with JNI
    data: *mut u8,
}

impl From<Vec<u8>> for ByteBuffer {
    fn from(other: Vec<u8>) -> Self {
        let mut buf = other.into_boxed_slice();
        let data = buf.as_mut_ptr();
        let len = buf.len();
        std::mem::forget(buf);
        Self {
            len: len as i64,
            data,
        }
    }
}

#[repr(C)]
pub struct AudioInfo {
    sample_rate: u32,
    num_channels: u32,
    sample_width: u32,
}

#[repr(C)]
pub struct SynthesisParams {
    mode: SynthesisMode,
    rate: u8,
    volume: u8,
    pitch: u8,
    appended_silence_ms: u32,
    callback: SpeechSynthesisCallback,
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
    fn as_piper_synth_config(&self) -> PiperSynthesisConfig {
        PiperSynthesisConfig {
            speaker: Some(self.speaker.into()),
            noise_scale: self.noise_scale,
            length_scale: self.length_scale,
            noise_w: self.noise_w,
        }
    }
}

#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataFreeString(string_ptr: *mut i8) {
    _internal_libsonataFreeString(string_ptr)
}

#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataFreeByteBuffer(buf: ByteBuffer) {
    let s = std::slice::from_raw_parts_mut(buf.data, buf.len as usize);
    drop(Box::from_raw(s as *mut [u8]));
}

#[no_mangle]
#[allow(non_snake_case)]
pub extern "C" fn libsonataLoadVoiceFromConfigPath(
    config_path_ptr: FfiStr,
    out_error: &mut ExternError,
) -> *mut SonataVoice {
    let config_path = config_path_ptr.into_string();
    call_with_result(out_error, move || _load_piper_voice(config_path))
}

#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataUnloadSonataVoice(voice_ptr: *mut SonataVoice) {
    _internal_libsonataUnloadSonataVoice(voice_ptr)
}

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

#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn libsonataSpeak(
    voice_ptr: *mut SonataVoice,
    text_ptr: FfiStr,
    params: SynthesisParams,
    out_error: &mut ExternError,
) {
    let voice = voice_ptr.as_ref().unwrap();
    let text = text_ptr.into_string();
    call_with_result(out_error, move || _synthesize(voice, text, params))
}

fn get_ort_environment() -> &'static Arc<ort::Environment> {
    ORT_ENVIRONMENT.get_or_init(|| {
        Arc::new(
            ort::Environment::builder()
                .with_name("sonata")
                .with_execution_providers([ort::ExecutionProvider::CPU(Default::default())])
                .build()
                .unwrap(),
        )
    })
}

fn _load_piper_voice(config_path: String) -> SonataFFIResult<SonataVoice> {
    let config_path = PathBuf::from(config_path);
    let piper_model = sonata_piper::from_config_path(&config_path, get_ort_environment())?;
    let synth = SonataSpeechSynthesizer::new(piper_model)?;
    Ok(synth.into())
}

fn _synthesize(
    voice: &SonataVoice,
    text: String,
    params: SynthesisParams,
) -> SonataFFIResult<()> {
    let synth: &SonataSpeechSynthesizer = &voice;
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
        SynthesisMode::BATCHED => {
            let stream = synth
                .synthesize_batched(text, audio_output_config, None)?
                .map(|wr| wr.map(|aud| aud.samples));
            iterate_stream(stream, params.callback)
        }
        SynthesisMode::REALTIME => {
            let synth: Arc<SonataSpeechSynthesizer> = Arc::clone(&voice.0);
            let (tx, rx) = channel();
            SYNTHESIS_THREAD_POOL.spawn(move || {
                let stream = synth.synthesize_streamed(text, audio_output_config, 72, 3).unwrap();
                stream.for_each(|result| {
                    tx.send(result).unwrap();
                });
            });
            iterate_stream(rx.into_iter(), params.callback)
        }
    }
}

#[inline(always)]
fn iterate_stream(
    stream: impl Iterator<Item = SonataResult<AudioSamples>>,
    callback: SpeechSynthesisCallback,
) -> SonataFFIResult<()> {
    for result in stream {
        let audio = result?;
        let wav_bytes = audio.as_wave_bytes();
        if !callback(wav_bytes.into()) {
            return Ok(());
        }
    }
    Ok(())
}
