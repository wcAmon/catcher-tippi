# Tippi for Windows — third-party notices

Tippi uses the following third-party components:

- NVIDIA Nemotron 3.5 ASR Streaming 0.6B, under the OpenMDW-1.1 license.
  The application downloads the pinned INT4 ONNX conversion directly from
  `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`; model files are
  not embedded in the application.
- sherpa-onnx 1.13.4, copyright the sherpa-onnx contributors, Apache-2.0.
  Tippi bundles its managed and Windows x64 CPU runtime libraries.
- Pyannote segmentation 3.0, copyright 2023 CNRS, MIT. The application
  downloads a pinned INT8 ONNX conversion; model files are not embedded.
- NVIDIA TitaNet-S, used for speaker embeddings. NVIDIA's model card states
  that its license is covered by the NVIDIA NeMo Toolkit license (Apache-2.0).
  The application downloads a pinned ONNX conversion; model files are not
  embedded.
- ONNX Runtime GenAI and ONNX Runtime, copyright Microsoft Corporation, MIT.
- DirectML and the D3D12 redistributable, copyright Microsoft Corporation,
  distributed under their accompanying Microsoft license terms.
- NAudio, copyright Mark Heath, MIT.
- OpenccNetLib, MIT; its bundled OpenCC dictionaries are Apache-2.0.
- Microsoft Visual C++ Redistributable files, distributed under the Microsoft
  Visual Studio license terms.

The complete ONNX Runtime, ONNX Runtime GenAI, DirectML/D3D12 redistributable,
NAudio, Pyannote segmentation, and Apache-2.0 license/notices are included in
the `Licenses` directory beside the app. `OpenCC-Dictionaries-LICENSE.txt`
contains the complete Apache-2.0 text that also applies to the bundled
sherpa-onnx runtime.

OpenccNetLib project license:
https://github.com/laisuk/OpenccNet/blob/master/OpenccNetLib/LICENSE

Model license and origin:
https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b

Speaker model origins and licenses:
https://huggingface.co/pyannote/segmentation-3.0
https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/titanet_small
https://github.com/NVIDIA-NeMo/Speech/blob/main/LICENSE

sherpa-onnx source and license:
https://github.com/k2-fsa/sherpa-onnx
