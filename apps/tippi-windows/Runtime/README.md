# Pinned Windows CPU runtime

The native binaries in `win-x64` are intentionally checked in so that a normal
build cannot silently fall back to ONNX Runtime GenAI 0.13, which cannot parse
Nemotron's multilingual `lang_id` input.

- ONNX Runtime GenAI source: `microsoft/onnxruntime-genai`
- Source commit: `e258b88a99edd00beaf00708393dbc54e31aacf1`
- Build options: `USE_CUDA=OFF`, `USE_DML=OFF`, `USE_WINML=OFF`
- sherpa-onnx managed/runtime package: `org.k2fsa.sherpa.onnx` and
  `org.k2fsa.sherpa.onnx.runtime.win-x64` version `1.13.4`
- sherpa-onnx source commit: `142807252687d81b40d6315f23470a1512a00de3`
- Shared ONNX Runtime dependency: official CPU package `1.27.0` from the
  sherpa-onnx Windows x64 runtime package
- Architecture: Windows x64
- MSVC redistributable: Visual C++ 2022 `14.44.35112`, copied beside the app

The source itself was unchanged. `/Qspectre` was disabled at compile time
because that optional library was not present in the local Build Tools install;
this does not enable a GPU provider or alter model behavior.

## SHA-256

```text
msvcp140.dll            0f885b509a685d2bbfa652fed26b5fb31d88fbdab0a978c641d1c7b8aa460aa9
msvcp140_1.dll          bfad5aef4c63a669e3c140655cdfdf395b6c979b400a447bd5dcb65ed8826c3d
onnxruntime-genai.dll   97ee417fa958a7607c1ba57e14b5e3febc383b3f74d394d5e8b636c165384209
onnxruntime.dll         daa77083a45bf525da0dde9e87f85d8eb146f58f9c9aa7124ca84545e1c0f148
sherpa-onnx.dll         9cef5904ac912106dfa8aaf0c70a4e5a86370fe08781f981d37cbd49e98fd37b
sherpa-onnx-c-api.dll   614878147c05121aeb1514ec4fb3e48b89751591532eca9208235b9ab868306a
vcruntime140.dll        d5e4d9a3e835fa679450145d6a7d94e36573a509317111904d9b3712c30d9066
vcruntime140_1.dll      1f2d41c4aa5db0bc33ebf7b66d72943a817d7ce6cbe880502a9403823633093f
```
