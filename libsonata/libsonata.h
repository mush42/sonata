/* Generated with cbindgen:0.26.0 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define FAILED_TO_LOAD_RESOURCE 17

#define PHONEMIZATION_ERROR 18

#define OPERATION_ERROR 19

#define INVALID_UTF8_SEQUENCE 20

#define UNKNOWN_ERROR 21

typedef enum SynthesisMode {
  LAZY = 0,
  PARALLEL = 1,
  REALTIME = 2,
} SynthesisMode;

typedef struct SonataVoice SonataVoice;

typedef struct PiperSynthConfig {
  uint32_t speaker;
  float length_scale;
  float noise_scale;
  float noise_w;
} PiperSynthConfig;

typedef int32_t ErrorCode;
#define ErrorCode_SUCCESS 0
#define ErrorCode_PANIC -1
#define ErrorCode_INVALID_HANDLE -1000

typedef struct ExternError {
  ErrorCode code;
  char *message;
} ExternError;

typedef struct LibsonataBuffer {
  int64_t len;
  uint8_t *data;
  struct ExternError *error_ptr;
} LibsonataBuffer;

typedef const char *FfiStr;

typedef struct AudioInfo {
  uint32_t sample_rate;
  uint32_t num_channels;
  uint32_t sample_width;
} AudioInfo;

typedef bool (*SpeechSynthesisCallback)(struct LibsonataBuffer);

typedef struct SynthesisParams {
  enum SynthesisMode mode;
  uint8_t rate;
  uint8_t volume;
  uint8_t pitch;
  uint32_t appended_silence_ms;
  SpeechSynthesisCallback callback;
  bool nonblocking;
} SynthesisParams;

void libsonataFreeString(int8_t *string_ptr);

void libsonataFreePiperSynthConfig(struct PiperSynthConfig *synth_config);

void libsonataFreeLibsonataBuffer(struct LibsonataBuffer buf);

struct SonataVoice *libsonataLoadVoiceFromConfigPath(FfiStr config_path_ptr,
                                                     struct ExternError *out_error);

void libsonataUnloadSonataVoice(struct SonataVoice *voice_ptr);

void libsonataGetAudioInfo(struct SonataVoice *voice_ptr,
                           struct AudioInfo *audio_info_ptr,
                           struct ExternError *out_error);

struct PiperSynthConfig *libsonataGetPiperDefaultSynthConfig(struct SonataVoice *voice_ptr,
                                                             struct ExternError *out_error);

void libsonataSetPiperSynthConfig(struct SonataVoice *voice_ptr,
                                  struct PiperSynthConfig synth_config,
                                  struct ExternError *out_error);

void libsonataSpeak(struct SonataVoice *voice_ptr,
                    FfiStr text_ptr,
                    struct SynthesisParams params,
                    struct ExternError *out_error);
