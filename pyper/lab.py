import os
from pyper import Piper, VitsModel, AudioOutputConfig



os.environ["ORT_DYLIB_PATH"] = r"D:\onnxruntime_libs\x86\onnxruntime.dll"
os.environ["PIPER_ESPEAKNG_DATA_DIRECTORY"] = r"D:\projects\blindpandas\piper-rs\deps\windows\espeak-ng-build"

vits = VitsModel(
    r"D:\Piper_TTS_Voices\voices\voice-en-us-amy-low\en-us-amy-low.onnx.json",
    r"D:\Piper_TTS_Voices\voices\voice-en-us-amy-low\en-us-amy-low.onnx"
)

p = Piper.with_vits(vits)


text = "Who are you? said the Caterpillar. Replied Alice , rather shyly, I hardly know, sir!"
p.synthesize_to_file(
    "output.wav",
    text,
    AudioOutputConfig(None, None, None, 1000),
)
