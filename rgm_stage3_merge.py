#!/usr/bin/env python3
"""Stage 3: solve RGM coefficients and write the merged checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rgm_config import defaults, load_setting, model_paths
from rgm_stage1_gram import EDGE_BOUNDARIES, NODE_ORDER, PER_LAYER_2D_PROJS


def load_stage2(stage2_dir: Path) -> tuple[dict, dict[tuple[int, str], dict]]:
    summary_path = stage2_dir / "summary.json"
    stats_path = stage2_dir / "node_statistics.jsonl"
    if not summary_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(
            f"Stage 3 requires {summary_path} and {stats_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = {}
    with stats_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[(int(row["layer"]), str(row["node"]))] = row
    return summary, rows


def solve_system(gamma: np.ndarray, ridge: float) -> dict:
    gamma = np.asarray(gamma, dtype=np.float64)
    diagonal = np.diag(gamma).copy()
    positive = diagonal[diagonal > 0]
    scale = float(np.mean(positive)) if len(positive) else 1.0
    matrix = gamma + ridge * scale * np.eye(len(diagonal), dtype=np.float64)
    try:
        coeffs = np.linalg.solve(matrix, diagonal)
        solve_ok = True
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(matrix, diagonal, rcond=None)[0]
        solve_ok = False
    retention = gamma @ coeffs / (diagonal + 1e-30)
    eigenvalues = np.linalg.eigvalsh(gamma)
    threshold = max(scale * 1e-12, 1e-30)
    positive_eigenvalues = eigenvalues[eigenvalues > threshold]
    condition = (
        float(positive_eigenvalues[-1] / positive_eigenvalues[0])
        if len(positive_eigenvalues) >= 2
        else None
    )
    return {
        "coeffs": coeffs,
        "retention": retention,
        "eig": eigenvalues,
        "cond": condition,
        "solve_ok": solve_ok,
    }


def mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def write_coefficient_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# RGM Coefficient Summary\n\n",
        f"- branch: `{summary['branch']}`\n",
        f"- experts: `{summary['experts']}`\n",
        f"- layers: `{summary['layers']}`\n",
        f"- ridge: `{summary['ridge']}`\n\n",
        "## Node coefficients\n\n",
        "| Layer | Node | Coefficients | Retention | Condition |\n",
        "| ---: | --- | --- | ---: | ---: |\n",
    ]
    for row in summary["node_rows"]:
        lines.append(
            f"| {row['layer']} | {row['node']} | "
            f"{[round(value, 6) for value in row['coeffs']]} | "
            f"{[round(value, 6) for value in row['retention']]} | "
            f"{row['cond']} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    if single_path.exists():
        with safe_open(str(single_path), framework="pt", device="cpu") as handle:
            return {name: "model.safetensors" for name in handle.keys()}
    raise FileNotFoundError(
        f"no safetensors index or single model.safetensors found in {model_dir}"
    )


def is_per_layer_2d(name: str) -> bool:
    return name.startswith("model.layers.") and any(
        name.endswith(suffix) for suffix in PER_LAYER_2D_PROJS
    )


def load_edge_coeffs(path: Path) -> dict[str, list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["tensor"]: [float(value) for value in row["edge_coeffs"]]
        for row in data["edge_rows"]
    }


def merge_raw_edge(
    base_tensor: torch.Tensor,
    expert_tensors: list[torch.Tensor],
    coeffs: list[float],
) -> torch.Tensor:
    result = base_tensor.float().clone()
    base_float = base_tensor.float()
    for coeff, expert_tensor in zip(coeffs, expert_tensors):
        result.add_(expert_tensor.float() - base_float, alpha=float(coeff))
    return result


def merge_taskmean(
    base_tensor: torch.Tensor,
    expert_tensors: list[torch.Tensor],
) -> torch.Tensor:
    base_float = base_tensor.float()
    delta = torch.zeros_like(base_float)
    for expert_tensor in expert_tensors:
        delta.add_(expert_tensor.float() - base_float, alpha=1.0 / len(expert_tensors))
    return base_float + delta


def same_shape_as_base(
    base_tensor: torch.Tensor,
    expert_tensors: list[torch.Tensor],
) -> bool:
    return all(tuple(tensor.shape) == tuple(base_tensor.shape) for tensor in expert_tensors)


def load_vocab(model_dir: Path) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    return tokenizer.get_vocab()


def remap_rows_to_base_vocab(
    base_vocab: dict[str, int],
    source_vocab: dict[str, int],
    source_weight: torch.Tensor,
    base_shape: torch.Size,
) -> torch.Tensor:
    if source_weight.dim() != 2:
        raise ValueError(f"expected a 2D vocabulary tensor, got {source_weight.shape}")
    result = torch.zeros(base_shape, dtype=source_weight.dtype)
    for token, base_id in base_vocab.items():
        source_id = source_vocab.get(token)
        if (
            source_id is not None
            and source_id < source_weight.shape[0]
            and base_id < result.shape[0]
        ):
            result[base_id] = source_weight[source_id]
    return result


def load_vocab_aligned_experts(
    name: str,
    base_tensor: torch.Tensor,
    expert_dirs: list[Path],
    expert_indices: list[dict[str, str]],
    base_vocab: dict[str, int],
    expert_vocabs: list[dict[str, int]],
) -> tuple[list[torch.Tensor], bool]:
    tensors = []
    remapped = False
    for expert_dir, expert_index, expert_vocab in zip(
        expert_dirs, expert_indices, expert_vocabs
    ):
        with safe_open(str(expert_dir / expert_index[name]), framework="pt") as handle:
            tensor = handle.get_tensor(name)
        if tuple(tensor.shape) != tuple(base_tensor.shape):
            tensor = remap_rows_to_base_vocab(
                base_vocab, expert_vocab, tensor, base_tensor.shape
            )
            remapped = True
        tensors.append(tensor)
    return tensors, remapped


def copy_base_side_files(base_dir: Path, out_dir: Path) -> None:
    for path in base_dir.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors"):
            shutil.copy2(path, out_dir / path.name)


def merge_checkpoint(
    base: str,
    experts: list[str],
    edge_coeffs: dict[str, list[float]],
    out_dir: Path,
    out_dtype_name: str,
) -> None:
    base_dir = Path(base)
    expert_dirs = [Path(path) for path in experts]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[out_dtype_name]

    copy_base_side_files(base_dir, out_dir)
    base_index = load_weight_map(base_dir)
    expert_indices = [load_weight_map(path) for path in expert_dirs]
    vocab_names = [
        name
        for name in ("model.embed_tokens.weight", "lm_head.weight")
        if name in base_index and all(name in index for index in expert_indices)
    ]
    if vocab_names:
        base_vocab = load_vocab(base_dir)
        expert_vocabs = [load_vocab(path) for path in expert_dirs]
    else:
        base_vocab = {}
        expert_vocabs = []

    shards: dict[str, list[str]] = {}
    for name, shard in base_index.items():
        shards.setdefault(shard, []).append(name)
    missing = sorted(
        name
        for name in base_index
        if is_per_layer_2d(name) and name not in edge_coeffs
    )
    if missing:
        raise RuntimeError(
            f"missing coefficients for {len(missing)} layer tensors; first={missing[:3]}"
        )

    print(
        f"[stage3] experts={len(expert_dirs)} modeled_edges={len(edge_coeffs)} "
        f"tensors={len(base_index)} out_dtype={out_dtype_name}",
        flush=True,
    )
    coeff_log = []
    fallback_log = []
    started = time.time()
    seen = 0
    for shard_name in sorted(shards):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(base_dir / shard_name), framework="pt", device="cpu") as base_file:
            for name in sorted(shards[shard_name]):
                seen += 1
                base_tensor = base_file.get_tensor(name)
                if is_per_layer_2d(name):
                    coeffs = edge_coeffs[name]
                    expert_tensors = []
                    for expert_dir, expert_index in zip(expert_dirs, expert_indices):
                        with safe_open(
                            str(expert_dir / expert_index[name]),
                            framework="pt",
                            device="cpu",
                        ) as expert_file:
                            expert_tensors.append(expert_file.get_tensor(name))
                    merged = merge_raw_edge(base_tensor, expert_tensors, coeffs)
                    tensors[name] = merged.to(out_dtype).contiguous()
                    coeff_log.append((name, coeffs))
                elif base_tensor.dim() == 1 and all(
                    name in index for index in expert_indices
                ):
                    expert_tensors = []
                    for expert_dir, expert_index in zip(expert_dirs, expert_indices):
                        with safe_open(
                            str(expert_dir / expert_index[name]),
                            framework="pt",
                            device="cpu",
                        ) as expert_file:
                            expert_tensors.append(expert_file.get_tensor(name))
                    if same_shape_as_base(base_tensor, expert_tensors):
                        tensors[name] = merge_taskmean(base_tensor, expert_tensors).to(out_dtype).contiguous()
                        fallback_log.append((name, "1d_taskmean"))
                    else:
                        tensors[name] = base_tensor.to(out_dtype).contiguous()
                        fallback_log.append((name, "base_copy_shape_mismatch"))
                elif base_tensor.dim() == 2 and all(
                    name in index for index in expert_indices
                ):
                    if name in vocab_names:
                        expert_tensors, remapped = load_vocab_aligned_experts(
                            name,
                            base_tensor,
                            expert_dirs,
                            expert_indices,
                            base_vocab,
                            expert_vocabs,
                        )
                    else:
                        expert_tensors = []
                        remapped = False
                        for expert_dir, expert_index in zip(expert_dirs, expert_indices):
                            with safe_open(
                                str(expert_dir / expert_index[name]),
                                framework="pt",
                                device="cpu",
                            ) as expert_file:
                                expert_tensors.append(expert_file.get_tensor(name))
                    if same_shape_as_base(base_tensor, expert_tensors):
                        tensors[name] = merge_taskmean(base_tensor, expert_tensors).to(out_dtype).contiguous()
                        reason = (
                            "2d_vocab_aligned_taskmean"
                            if remapped
                            else "2d_unmodeled_taskmean"
                        )
                        fallback_log.append((name, reason))
                    else:
                        tensors[name] = base_tensor.to(out_dtype).contiguous()
                        fallback_log.append((name, "base_copy_shape_mismatch"))
                else:
                    tensors[name] = base_tensor.to(out_dtype).contiguous()
                    fallback_log.append((name, "base_copy"))
                if seen % 10 == 0:
                    print(f"[stage3 merge {seen}/{len(base_index)}] {name}", flush=True)
        save_file(tensors, str(out_dir / shard_name))
        print(f"[stage3] wrote {shard_name} ({len(tensors)} tensors)", flush=True)

    (out_dir / "metric_sheaf_edge_coefficients.json").write_text(
        json.dumps({"tensors": coeff_log}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "fallback_tensors.json").write_text(
        json.dumps({"fallbacks": fallback_log}, indent=2) + "\n",
        encoding="utf-8",
    )
    elapsed = (time.time() - started) / 60.0
    print(f"[stage3] checkpoint write-back complete in {elapsed:.1f} min", flush=True)


def compute_coefficients(
    config: dict,
    config_path: Path,
    stage2_dir: Path,
    out_dir: Path,
    branch_request: str,
    ridge: float,
) -> dict:
    gate_summary, rows = load_stage2(stage2_dir)
    if branch_request == "auto":
        branch = str(gate_summary["branch"])
    else:
        branch = branch_request
    if branch not in {"raw", "normbalanced"}:
        raise ValueError(f"invalid branch: {branch}")
    _, expert_paths, expert_names, layers = model_paths(config)
    expected_branch = "normbalanced" if gate_summary["decision"] == "NormBalanced-RGM" else "raw"
    if branch != expected_branch:
        print(
            f"[stage3] requested branch={branch} differs from gate={expected_branch}",
            flush=True,
        )

    node_rows = []
    coeff_by_layer_node: dict[tuple[int, str], list[float]] = {}
    for layer in layers:
        for node in NODE_ORDER:
            key = (int(layer), node)
            if key not in rows:
                raise KeyError(f"Stage 2 is missing node statistics for {key}")
            row = rows[key]
            matrix_name = "raw_gamma" if branch == "raw" else "normbalanced_gamma"
            solution = solve_system(np.asarray(row[matrix_name]), ridge)
            coeffs = [float(value) for value in solution["coeffs"]]
            retention = [float(value) for value in solution["retention"]]
            coeff_by_layer_node[key] = coeffs
            rho = np.asarray(row["rho"], dtype=np.float64)
            offdiag = [
                float(rho[i, j])
                for i in range(len(expert_names))
                for j in range(i + 1, len(expert_names))
            ]
            positive_competitions = []
            norms = [float(value) for value in row["metric_norms"]]
            for i in range(len(expert_names)):
                for j in range(i + 1, len(expert_names)):
                    if rho[i, j] > 0:
                        positive_competitions.extend(
                            [
                                float(rho[i, j]) * norms[j] / max(norms[i], 1e-30),
                                float(rho[i, j]) * norms[i] / max(norms[j], 1e-30),
                            ]
                        )
            node_rows.append(
                {
                    "layer": int(layer),
                    "node": node,
                    "coeffs": coeffs,
                    "retention": retention,
                    "cond": solution["cond"],
                    "solve_ok": solution["solve_ok"],
                    "mean_pair_cos": mean(offdiag),
                    "max_abs_pair_cos": max((abs(value) for value in offdiag), default=None),
                    "max_positive_competition": max(positive_competitions, default=0.0),
                    "metric_norm_ratio": (
                        max(norms) / max((value for value in norms if value > 0), default=1e-30)
                        if norms
                        else None
                    ),
                    "pair_cos_matrix": rho.tolist(),
                    "metric_norms": norms,
                }
            )

    edge_rows = []
    for layer in layers:
        for suffix, boundaries in EDGE_BOUNDARIES.items():
            tensor_name = f"model.layers.{layer}.{suffix}"
            node_coeff_lists = [
                coeff_by_layer_node[(layer, node)]
                for node, _ in boundaries
                if (layer, node) in coeff_by_layer_node
            ]
            if not node_coeff_lists:
                continue
            edge_coeffs = []
            for expert_index in range(len(expert_names)):
                values = [coeffs[expert_index] for coeffs in node_coeff_lists]
                product = 1.0
                for value in values:
                    product *= value
                edge_coeffs.append(product ** (1.0 / len(values)))
            edge_rows.append(
                {
                    "tensor": tensor_name,
                    "layer": int(layer),
                    "suffix": suffix,
                    "edge_coeffs": edge_coeffs,
                }
            )

    all_node_coeffs = [value for row in node_rows for value in row["coeffs"]]
    all_edge_coeffs = [value for row in edge_rows for value in row["edge_coeffs"]]
    summary = {
        "stage": "stage3_merge",
        "method": "RGM" if branch == "raw" else "RGM-NormBalanced",
        "branch": branch,
        "branch_request": branch_request,
        "gate_decision": gate_summary["decision"],
        "config": str(config_path),
        "stage2_dir": str(stage2_dir),
        "base": config["base"],
        "experts": expert_names,
        "expert_paths": expert_paths,
        "layers": layers,
        "ridge": ridge,
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "num_nodes": len(node_rows),
        "num_edges": len(edge_rows),
        "node_coeff_mean": mean(all_node_coeffs),
        "node_coeff_min": min(all_node_coeffs) if all_node_coeffs else None,
        "node_coeff_max": max(all_node_coeffs) if all_node_coeffs else None,
        "edge_coeff_mean": mean(all_edge_coeffs),
        "edge_coeff_min": min(all_edge_coeffs) if all_edge_coeffs else None,
        "edge_coeff_max": max(all_edge_coeffs) if all_edge_coeffs else None,
    }
    coefficients_dir = out_dir / "coefficients"
    coefficients_dir.mkdir(parents=True, exist_ok=True)
    (coefficients_dir / "rgm_coeff_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (coefficients_dir / "metric_sheaf_coeff_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_coefficient_markdown(summary, coefficients_dir / "rgm_coeff_summary.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: solve branch coefficients and write the merged checkpoint."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage2-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--branch",
        choices=("auto", "raw", "normbalanced"),
        default="auto",
    )
    parser.add_argument("--ridge", type=float, default=None)
    parser.add_argument("--out-dtype", choices=("bfloat16", "float16", "float32"), default=None)
    args = parser.parse_args()

    config = load_setting(args.config, validate_paths=True)
    values = defaults(config)
    ridge = float(values["ridge"] if args.ridge is None else args.ridge)
    out_dtype = str(values["out_dtype"] if args.out_dtype is None else args.out_dtype)
    config_path = args.config.expanduser().resolve()
    stage2_dir = args.stage2_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    summary = compute_coefficients(
        config,
        config_path,
        stage2_dir,
        out_dir,
        args.branch,
        ridge,
    )
    edge_coeffs = {
        row["tensor"]: row["edge_coeffs"] for row in summary["edge_rows"]
    }
    base, experts, _, _ = model_paths(config)
    merge_checkpoint(
        base,
        experts,
        edge_coeffs,
        out_dir / "merged_model",
        out_dtype,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                **summary,
                "out_dtype": out_dtype,
                "merged_model": str(out_dir / "merged_model"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[stage3] completed branch={summary['branch']} output={out_dir}", flush=True)


if __name__ == "__main__":
    main()
