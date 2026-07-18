# Pinned Windows CPU / DirectML runtime

The native binaries in `win-x64` are intentionally checked in so that a normal
build cannot silently fall back to ONNX Runtime GenAI 0.13, which cannot parse
Nemotron's multilingual `lang_id` input.

- ONNX Runtime GenAI source: `microsoft/onnxruntime-genai`
- Source commit: `e258b88a99edd00beaf00708393dbc54e31aacf1`
- Build options: `USE_CUDA=OFF`, `USE_DML=ON`, `USE_WINML=OFF`
- ONNX Runtime DirectML nightly package:
  `1.29.0-dev-20260714-1047-a91b0b49cb`, source commit
  `a91b0b49cb0dc9670a8cf93263b3d79ce0dc79a5`
- DirectML redistributable: `1.15.4`
- D3D12 redistributable: `1.614.1`
- sherpa-onnx managed/runtime package: `org.k2fsa.sherpa.onnx` and
  `org.k2fsa.sherpa.onnx.runtime.win-x64` version `1.13.4`
- sherpa-onnx source commit: `142807252687d81b40d6315f23470a1512a00de3`
- Shared ONNX Runtime dependency: the pinned 1.29 DirectML build above. It is
  backward-compatible with the ORT API 27 requested by sherpa-onnx.
- Architecture: Windows x64
- MSVC redistributable: Visual C++ 2022 `14.44.35112`, copied beside the app

Only build-system flags were changed: `/MP` was enabled to parallelize MSVC and
`/Qspectre` was disabled because its optional library was not installed. No
ONNX Runtime GenAI inference source was changed.

VoxCPM2 uses a separate process and separate ggml runtime so its Vulkan backend
cannot conflict with this ONNX Runtime/DirectML stack. See
[`tts/README.md`](tts/README.md) for that runtime's pinned source and hashes.

The shipped INT4 Nemotron graph currently fails a real-audio DirectML probe on
the tested Intel Iris Xe and NVIDIA RTX 4060 Laptop GPUs. Tippi therefore
automatically selects CPU for this pinned model revision. DirectML remains in
the runtime so that a compatible future model revision can be selected without
changing the application architecture; a decode failure always replays the
session on CPU.

## SHA-256

```text
D3D12Core.dll           8a23d826b25b4329522ff451cb52b7f2b34d7f2913cfeb878371ce8bd765fe2d
DirectML.dll            9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1
msvcp140.dll            0f885b509a685d2bbfa652fed26b5fb31d88fbdab0a978c641d1c7b8aa460aa9
msvcp140_1.dll          bfad5aef4c63a669e3c140655cdfdf395b6c979b400a447bd5dcb65ed8826c3d
onnxruntime-genai.dll   7b34b5856b1b0b5d8590be37300fe6224169f220a6708e51018b1f90b1dfc3b7
onnxruntime.dll         cb0380c4072a32d1e2a1aeda9d54b94c4f645df9f81e9b37535559e57938c908
sherpa-onnx.dll         9cef5904ac912106dfa8aaf0c70a4e5a86370fe08781f981d37cbd49e98fd37b
sherpa-onnx-c-api.dll   614878147c05121aeb1514ec4fb3e48b89751591532eca9208235b9ab868306a
vcruntime140.dll        d5e4d9a3e835fa679450145d6a7d94e36573a509317111904d9b3712c30d9066
vcruntime140_1.dll      1f2d41c4aa5db0bc33ebf7b66d72943a817d7ce6cbe880502a9403823633093f
```
