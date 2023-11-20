/* Generated with cbindgen:0.26.0 */

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

constexpr static const int32_t FAILED_TO_LOAD_RESOURCE = 17;

constexpr static const int32_t PHONEMIZATION_ERROR = 18;

constexpr static const int32_t OPERATION_ERROR = 19;

constexpr static const int32_t UNKNOWN_ERROR = 21;

enum class SynthesisMode {
  LAZY = 0,
  PARALLEL = 1,
  BATCHED = 2,
  REALTIME = 3,
};

struct SonataVoice;

struct ByteBuffer {
  int64_t len;
  uint8_t *data;
};

using FfiStr = const char*;

using ErrorCode = int32_t;
constexpr static const ErrorCode ErrorCode_SUCCESS = 0;
constexpr static const ErrorCode ErrorCode_PANIC = -1;
constexpr static const ErrorCode ErrorCode_INVALID_HANDLE = -1000;

struct ExternError {
  ErrorCode code;
  char *message;
};

struct AudioInfo {
  uint32_t sample_rate;
  uint32_t num_channels;
  uint32_t sample_width;
};

struct PiperSynthConfig {
  uint32_t speaker;
  float length_scale;
  float noise_scale;
  float noise_w;
};

using SpeechSynthesisCallback = bool(*)(ByteBuffer);

struct SynthesisParams {
  SynthesisMode mode;
  uint8_t rate;
  uint8_t volume;
  uint8_t pitch;
  uint32_t appended_silence_ms;
  SpeechSynthesisCallback callback;
};

extern "C" {

void libsonataFreeString(int8_t *string_ptr);

void libsonataFreeByteBuffer(ByteBuffer buf);

SonataVoice *libsonataLoadVoiceFromConfigPath(FfiStr config_path_ptr, ExternError *out_error);

void libsonataUnloadSonataVoice(SonataVoice *voice_ptr);

void libsonataGetAudioInfo(SonataVoice *voice_ptr,
                           AudioInfo *audio_info_ptr,
                           ExternError *out_error);

void libsonataSetPiperSynthConfig(SonataVoice *voice_ptr,
                                  PiperSynthConfig synth_config,
                                  ExternError *out_error);

void libsonataSpeak(SonataVoice *voice_ptr,
                    FfiStr text_ptr,
                    SynthesisParams params,
                    ExternError *out_error);

} // extern "C"
