import os
from pyper import Piper, SynthConfig, AudioOutputConfig



os.environ["ORT_DYLIB_PATH"] = r"D:\onnxruntime_libs\x86\onnxruntime.dll"
os.environ["PIPER_ESPEAKNG_DATA_DIRECTORY"] = r"D:\projects\blindpandas\piper-rs\deps\windows\espeak-ng-build"


p = Piper(
    r"D:\Piper_TTS_Voices\voices\voice-en-us-amy-low\en-us-amy-low.onnx.json",
    r"D:\Piper_TTS_Voices\voices\voice-en-us-amy-low\en-us-amy-low.onnx"
)


text = "Who are you? said the Caterpillar. Replied Alice , rather shyly, I hardly know, sir!"
p.synthesize_to_file(
    "output.wav",
    text,
    None,
    AudioOutputConfig(None, None, None, 1000),
)
