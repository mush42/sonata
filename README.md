# Piper-rs

A Rust frontend for [piper](https://github.com/rhasspy/piper).

# Packages

- `espeak-phonemizer`: Converts text to `IPA` phonemes using a patched version of eSpeak-ng
- `piper-model`: Handles Piper model loading and inference using `onnxruntime` via `ort`
- `piper-synth`: Adds additional functionality on top of `piper-model` such as controlling rate, volume, and pitch
- `pyper`: Python bindings to `piper-synth` using `pyo3`
- `sonic-sys`: Rust FFI bindings to [Sonic](https://github.com/waywardgeek/sonic): a `C` library for controlling various aspects of generated speech, such as rate, volume, and pitch

# A note on testing

Some packages, such as `espeak-phonemizer`, include tests. Running `cargo test` from the root of the workspace will likely fail, because `cargo` does not load `config` from sub packages when ran from the workspace root.

To properly run tests, `cd` to the package you want to test, and run `cargo test` from there.

# License

Copyright (c) 2023 Musharraf Omer. This code is licensed under the  MIT license.

