#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
ROOT=${SCRIPT_DIR:h:h:h}
MODEL=${1:-${ROOT}/../../.model-artifact}

cd ${ROOT}
NEMOTRON_MLX_ARTIFACT=${MODEL} cargo test -p catcher-ffi --test ffi_lifecycle \
    c_abi_transcribes_reference_wav_exactly -- --ignored --nocapture --test-threads=1
