import json
import math
import sys
from pathlib import Path

import torch
from transformers import Nemotron3_5AsrForRNNT


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: generate_encoder_reference.py MODEL_DIR OUTPUT_JSON")
    model_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    model = Nemotron3_5AsrForRNNT.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    values = [math.sin(index * 0.013) * 0.1 for index in range(25 * 128)]
    features = torch.tensor(values, dtype=torch.float32).reshape(1, 25, 128)
    attention_mask = torch.ones((1, 25), dtype=torch.long)
    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().float().cpu()

        return hook

    handles = [model.encoder.subsampling.register_forward_hook(capture("subsampling"))]
    handles.extend(
        layer.register_forward_hook(capture(f"layer_{index}"))
        for index, layer in enumerate(model.encoder.layers)
    )

    with torch.no_grad():
        encoder_outputs = model.encoder(
            input_features=features,
            attention_mask=attention_mask,
            use_cache=True,
            num_lookahead_tokens=3,
        )
        encoded = encoder_outputs.last_hidden_state
        one_hot = torch.nn.functional.one_hot(
            torch.tensor([101]), num_classes=model.config.num_prompts
        ).to(encoded.dtype)
        one_hot = one_hot[:, None, :].expand(-1, encoded.shape[1], -1)
        prompted = model.prompt_projector(torch.cat([encoded, one_hot], dim=-1))
    for handle in handles:
        handle.remove()

    def summary(tensor):
        flat = tensor.flatten()
        return {
            "shape": list(tensor.shape),
            "mean": flat.mean().item(),
            "rms": flat.square().mean().sqrt().item(),
            "first": flat[:64].tolist(),
        }

    checkpoints = {name: summary(tensor) for name, tensor in captured.items()}
    checkpoints["prompted"] = summary(prompted)

    second_values = [math.cos(index * 0.017) * 0.1 for index in range(32 * 128)]
    second_features = torch.tensor(second_values, dtype=torch.float32).reshape(1, 32, 128)
    second_captured = {}

    def capture_second(name):
        def hook(_module, _inputs, output):
            second_captured[name] = output.detach().float().cpu()

        return hook

    second_handles = [
        model.encoder.subsampling.register_forward_hook(capture_second("subsampling"))
    ]
    second_handles.extend(
        layer.register_forward_hook(capture_second(f"layer_{index}"))
        for index, layer in enumerate(model.encoder.layers)
    )
    with torch.no_grad():
        second_encoder_outputs = model.encoder(
            input_features=second_features,
            attention_mask=torch.ones((1, 32), dtype=torch.long),
            past_key_values=encoder_outputs.past_key_values,
            padding_cache=encoder_outputs.padding_cache,
            use_cache=True,
            num_lookahead_tokens=3,
        )
        second_encoded = second_encoder_outputs.last_hidden_state
        second_one_hot = one_hot[:, :1, :].expand(-1, second_encoded.shape[1], -1)
        second_prompted = model.prompt_projector(
            torch.cat([second_encoded, second_one_hot], dim=-1)
        )
    for handle in second_handles:
        handle.remove()
    second_checkpoints = {
        name: summary(tensor) for name, tensor in second_captured.items()
    }
    second_checkpoints["prompted"] = summary(second_prompted)
    with torch.no_grad():
        generated = model.generate(
            input_features=(chunk for chunk in [features, second_features]),
            prompt_ids=torch.tensor([101]),
            num_lookahead_tokens=3,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "input_shape": list(features.shape),
                "input": values,
                "output_shape": list(prompted.shape),
                "output": prompted.flatten().tolist(),
                "output_attention_mask": encoder_outputs.attention_mask.flatten().tolist(),
                "checkpoints": checkpoints,
                "second_input_shape": list(second_features.shape),
                "second_input": second_values,
                "second_output_shape": list(second_prompted.shape),
                "second_output": second_prompted.flatten().tolist(),
                "second_checkpoints": second_checkpoints,
                "generated_token_ids": generated.sequences.flatten().tolist(),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
