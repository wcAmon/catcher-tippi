"""Unpack a NeMo Sortformer checkpoint into F32 safetensors plus metadata.

Usage: python tools/export_sortformer_weights.py MODEL.nemo OUTPUT_DIR
Requires: pip install torch safetensors pyyaml   (no NeMo needed)
"""

import json
import sys
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

SKIP_SUFFIXES = ("num_batches_tracked",)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_sortformer_weights.py MODEL.nemo OUTPUT_DIR")
    nemo_path = Path(sys.argv[1])
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(nemo_path) as tar:
            tar.extractall(tmp, filter="data")
        tmp_path = Path(tmp)
        config_path = next(tmp_path.rglob("model_config.yaml"))
        ckpt_path = next(tmp_path.rglob("*.ckpt"))
        config = yaml.safe_load(config_path.read_text())
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    tensors: dict[str, torch.Tensor] = {}
    inventory: dict[str, list[int]] = {}
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor) or name.endswith(SKIP_SUFFIXES):
            continue
        converted = tensor.detach().to(torch.float32).contiguous()
        tensors[name] = converted
        inventory[name] = list(converted.shape)

    save_file(tensors, str(output / "model.safetensors"))
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    parameters = sum(t.numel() for t in tensors.values())
    print(f"exported {len(tensors)} tensors, {parameters:,} parameters")


if __name__ == "__main__":
    main()
