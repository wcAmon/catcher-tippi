# Pinned VoxCPM2 Windows runtime

The two subdirectories contain standalone `llama-tts-server` builds. Tippi
starts the server on loopback only and keeps it in a separate process so the
Nemotron ASR, sherpa-onnx keyword spotter, and speaker diarizer can continue
running on CPU.

- Source: `tc-mb/llama.cpp-omni`
- Source commit: `b9d15b83ee353b2eaeee4d9318c98a35a1347486`
- Build number reported by the executable: `257 (b9d15b8)`
- Compiler: MSVC 19.44.35228, Windows AMD64, Release
- Common options: `GGML_NATIVE=OFF`, `GGML_CUDA=OFF`, `LLAMA_CURL=OFF`,
  `LLAMA_BUILD_TESTS=OFF`, `LLAMA_BUILD_EXAMPLES=OFF`
- CPU options: `GGML_VULKAN=OFF`
- Vulkan options: `GGML_VULKAN=ON`, Vulkan SDK 1.4.350.0
- CPU ISA selected by upstream ggml on this x64 build: AVX2/FMA/F16C/BMI2
- OpenSSL is intentionally absent. The app uses plain HTTP bound only to
  `127.0.0.1`; the server does not make outbound HTTP requests.

The application does not bundle model files. It downloads and verifies only
`VoxCPM2-BaseLM-Q8_0.gguf` and `VoxCPM2-Acoustic-F16.gguf` from the pinned
GGUF repository revision.

## SHA-256

```text
cpu/ggml.dll                  54cab1e68c430df3166e9f56b0deca321df37b86fa83405cb75f74105137973e
cpu/ggml-base.dll             e584ca52e5395270b788c5670a1c8ff03add5f9b000692d9fc06c17846dac74c
cpu/ggml-cpu.dll              ebb1c9bf0ac1e264b6663abf54e9675b7770429a8ff1181f59732b1c8011525d
cpu/llama.dll                 446f95a70ea0977f57c2a8347cbd495fcf186c2b161c59e12f96d5316d53dc62
cpu/llama-common.dll          697e0b72bc28d9371feced5274f1a70883d10b873200cdf6a0aef50d56305400
cpu/llama-tts-server.exe      c6d561956f4c20f4c486533aa62cc0a39935aea7dfa8964f1922104620cebd57
vulkan/ggml.dll               e4c1d6665ce3d2e28840c8f47b1f7680a55feea1d553ecdad856edf0235c24ef
vulkan/ggml-base.dll          8f2a859d3741890ba196bcfa7b893b3ae32b22055c13d7e182264f5736fa8b97
vulkan/ggml-cpu.dll           7e625e43d02e2886ece3c7c40ac474d80c57c06cde49655b6156109fc31587c3
vulkan/ggml-vulkan.dll        9d7c26c47fb375e473bc9b72df5b6cd52196915cfc24e153feb5b2a972b39017
vulkan/llama.dll              79e56546c93b4d51aa65fc3b43204f2786fd2a428b0eb512c46a036a0959d907
vulkan/llama-common.dll       87711c0f2d6d988422c3cf7987f40d7c54d520b34e2fc54a521ddbd6c138a08e
vulkan/llama-tts-server.exe   5287998670c798509d14f69a10000859aaa9cc2c9e2b4e1e87e00405173a6993
```
