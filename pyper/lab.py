import os
from pyper import Piper, VitsModel, AudioOutputConfig



os.environ["ORT_DYLIB_PATH"] = r"C:\Users\ibnom\Downloads\onnxruntime-win-x64-1.16.1\lib\onnxruntime.dll"
os.environ["PIPER_ESPEAKNG_DATA_DIRECTORY"] = r"D:\projects\blindpandas\piper-rs\deps\windows\espeak-ng-build"

vits = VitsModel(
    r"C:\Users\ibnom\AppData\Roaming\nvda\piper\voices\v1.0\en_US-hfc_male_streaming-medium\en_US-hfc_male-medium.onnx.json",
)

p = Piper.with_vits(vits)


text = "Who are you? said the Caterpillar. Replied Alice , rather shyly, I hardly know, sir!"
p.synthesize_to_file(
    "output.wav",
    text,
    AudioOutputConfig(None, None, None, 0),
)
