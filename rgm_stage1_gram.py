#!/usr/bin/env python3
"""Stage 1: compute and store state-level Gram-shift sections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rgm_config import load_setting, model_paths


PER_LAYER_2D_PROJS: tuple[str, ...] = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

NODE_ORDER: tuple[str, ...] = ("attn_in", "post_attn", "post_mlp")

NODE_INCIDENT: dict[str, list[tuple[str, str]]] = {
    "attn_in": [
        ("right", "self_attn.q_proj.weight"),
        ("right", "self_attn.k_proj.weight"),
        ("right", "self_attn.v_proj.weight"),
    ],
    "post_attn": [
        ("left", "self_attn.o_proj.weight"),
        ("right", "mlp.gate_proj.weight"),
        ("right", "mlp.up_proj.weight"),
    ],
    "post_mlp": [
        ("left", "mlp.down_proj.weight"),
    ],
}

EDGE_BOUNDARIES: dict[str, list[tuple[str, str]]] = {
    "self_attn.q_proj.weight": [("attn_in", "right")],
    "self_attn.k_proj.weight": [("attn_in", "right")],
    "self_attn.v_proj.weight": [("attn_in", "right")],
    "self_attn.o_proj.weight": [("post_attn", "left")],
    "mlp.gate_proj.weight": [("post_attn", "right")],
    "mlp.up_proj.weight": [("post_attn", "right")],
    "mlp.down_proj.weight": [("post_mlp", "left")],
}


def section_key(node: str, expert_index: int) -> str:
    return f"{node}__expert_{expert_index:02d}"


class TensorLoader:
    """Lazy CPU safetensors loader used while constructing Stage 1 sections."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        single_path = self.model_dir / "model.safetensors"
        if index_path.exists():
            self.index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        elif single_path.exists():
            with safe_open(str(single_path), framework="pt", device="cpu") as handle:
                self.index = {name: "model.safetensors" for name in handle.keys()}
        else:
            raise FileNotFoundError(
                f"no safetensors index or single model.safetensors found in {self.model_dir}"
            )
        self.cache = {}

    def get(self, name: str, device: str = "cuda") -> torch.Tensor:
        shard = self.index[name]
        if shard not in self.cache:
            self.cache[shard] = safe_open(
                str(self.model_dir / shard), framework="pt", device="cpu"
            )
        return self.cache[shard].get_tensor(name).to(device=device, dtype=torch.float32)


def gram_shift_right(w0: torch.Tensor, wi: torch.Tensor) -> tuple[torch.Tensor, float]:
    delta = wi - w0
    section = w0.T @ delta + delta.T @ w0 + delta.T @ delta
    section = 0.5 * (section + section.T)
    delta_norm_sq = float(torch.sum(delta.double() ** 2).cpu())
    return section, delta_norm_sq


def gram_shift_left(w0: torch.Tensor, wi: torch.Tensor) -> tuple[torch.Tensor, float]:
    delta = wi - w0
    section = w0 @ delta.T + delta @ w0.T + delta @ delta.T
    section = 0.5 * (section + section.T)
    delta_norm_sq = float(torch.sum(delta.double() ** 2).cpu())
    return section, delta_norm_sq


def node_gram_shift(
    base_loader: TensorLoader,
    expert_loader: TensorLoader,
    layer: int,
    incident: list[tuple[str, str]],
    device: str,
) -> tuple[torch.Tensor, float]:
    aggregate = None
    delta_norm_sq = 0.0
    for side, suffix in incident:
        name = f"model.layers.{layer}.{suffix}"
        w0 = base_loader.get(name, device)
        wi = expert_loader.get(name, device)
        if side == "right":
            section, d2 = gram_shift_right(w0, wi)
        elif side == "left":
            section, d2 = gram_shift_left(w0, wi)
        else:
            raise ValueError(f"unknown Gram-shift side: {side}")
        aggregate = section if aggregate is None else aggregate + section
        delta_norm_sq += d2
        del w0, wi, section
    if aggregate is None:
        raise ValueError("empty incident set")
    return 0.5 * (aggregate + aggregate.T), delta_norm_sq


def compute_and_save(
    base: str,
    experts: list[str],
    expert_names: list[str],
    layers: list[int],
    device: str,
    out_dir: Path,
) -> dict:
    """Compute m_(v,i) and save one safetensors file per transformer layer."""
    base_loader = TensorLoader(base)
    expert_loaders = [TensorLoader(path) for path in experts]
    sections_dir = out_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    section_shapes: dict[str, list[int]] = {}
    section_files: dict[str, str] = {}
    for layer in layers:
        print(f"[stage1] layer {layer}", flush=True)
        layer_tensors: dict[str, torch.Tensor] = {}
        for node in NODE_ORDER:
            incident = NODE_INCIDENT[node]
            for expert_index, expert_loader in enumerate(expert_loaders):
                section, _ = node_gram_shift(
                    base_loader,
                    expert_loader,
                    layer,
                    incident,
                    device,
                )
                key = section_key(node, expert_index)
                section = section.detach().cpu().contiguous()
                layer_tensors[key] = section
                section_shapes[key] = list(section.shape)

                del section
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

        filename = f"layer_{layer:02d}.safetensors"
        save_file(layer_tensors, str(sections_dir / filename))
        section_files[str(layer)] = f"sections/{filename}"
        del layer_tensors

    summary = {
        "base": base,
        "experts": expert_names,
        "expert_paths": experts,
        "layers": layers,
        "nodes": list(NODE_ORDER),
        "section_files": section_files,
        "section_shapes": section_shapes,
        "num_layers": len(layers),
        "num_nodes": len(layers) * len(NODE_ORDER),
        "num_sections": len(layers) * len(NODE_ORDER) * len(experts),
        "dtype": "float32",
        "stage": "stage1_gram",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: compute state-level Gram-shift sections m_(v,i)."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_setting(args.config, validate_paths=True)
    base, experts, expert_names, layers = model_paths(config)
    summary = compute_and_save(
        base,
        experts,
        expert_names,
        layers,
        args.device,
        args.out_dir.expanduser().resolve(),
    )
    print(
        f"[stage1] wrote {summary['num_sections']} sections to "
        f"{args.out_dir.expanduser().resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
