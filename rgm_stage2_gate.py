#!/usr/bin/env python3
"""Stage 2: compute section statistics and choose the coefficient branch."""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from safetensors import safe_open

from rgm_config import defaults, load_setting, model_paths
from rgm_stage1_gram import NODE_ORDER, section_key


EPS = 1e-30
PAIR_COLORS = ("#7A5195", "#2E86AB", "#E69F00")
EXPERT_COLORS = ("#0072B2", "#D55E00", "#009E73")


def gamma_from_sections(sections: list) -> np.ndarray:
    k = len(sections)
    gamma = np.zeros((k, k), dtype=np.float64)
    doubled = [section.detach().double().reshape(-1) for section in sections]
    for i in range(k):
        for j in range(i, k):
            value = float((doubled[i] * doubled[j]).sum().cpu())
            gamma[i, j] = value
            gamma[j, i] = value
    return gamma


def section_statistics(gamma: np.ndarray, norm_eps: float) -> dict:
    diagonal = np.diag(gamma).astype(np.float64)
    norms = np.sqrt(np.maximum(diagonal, 0.0))
    rho = gamma / np.maximum(norms[:, None] * norms[None, :], EPS)
    np.fill_diagonal(rho, 1.0)

    d_by_expert = []
    for i in range(len(norms)):
        ni = max(float(norms[i]), EPS)
        d_by_expert.append(
            sum(
                max(float(rho[i, j]), 0.0) * max(float(norms[j]), EPS) / ni
                for j in range(len(norms))
                if j != i
            )
        )

    normbalanced_gamma = gamma / (
        (norms + norm_eps)[:, None] * (norms + norm_eps)[None, :]
    )
    return {
        "raw_gamma": gamma,
        "normbalanced_gamma": normbalanced_gamma,
        "metric_norms": norms,
        "rho": rho,
        "D_by_expert": np.asarray(d_by_expert, dtype=np.float64),
        "D_v": max(d_by_expert) if d_by_expert else 0.0,
    }


def load_layer_sections(
    stage1_dir: Path,
    layer: int,
    expert_count: int,
) -> dict[str, list]:
    path = stage1_dir / "sections" / f"layer_{layer:02d}.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing Stage 1 section file: {path}")
    loaded: dict[str, list] = {node: [] for node in NODE_ORDER}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for node in NODE_ORDER:
            for expert_index in range(expert_count):
                key = section_key(node, expert_index)
                if key not in handle.keys():
                    raise KeyError(f"{path}: missing tensor {key}")
                loaded[node].append(handle.get_tensor(key))
    return loaded


def compute_rows(
    stage1_dir: Path,
    expert_names: list[str],
    layers: list[int],
    norm_eps: float,
) -> list[dict]:
    rows = []
    for layer in layers:
        print(f"[stage2] layer {layer}", flush=True)
        sections = load_layer_sections(stage1_dir, layer, len(expert_names))
        for node in NODE_ORDER:
            stats = section_statistics(gamma_from_sections(sections[node]), norm_eps)
            rows.append(
                {
                    "layer": int(layer),
                    "node": node,
                    "label": f"L{layer:02d}/{node}",
                    "raw_gamma": stats["raw_gamma"].tolist(),
                    "normbalanced_gamma": stats["normbalanced_gamma"].tolist(),
                    "metric_norms": stats["metric_norms"].tolist(),
                    "rho": stats["rho"].tolist(),
                    "D_by_expert": stats["D_by_expert"].tolist(),
                    "D_v": float(stats["D_v"]),
                }
            )
    return rows


def finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def symmetric_limit(values: list[float]) -> float:
    finite = finite_values(values)
    peak = max((abs(value) for value in finite), default=0.0)
    step = 0.005 if peak <= 0.025 else 0.01
    return max(step * 2, math.ceil(peak / step) * step)


def render_raw_figures(
    rows: list[dict],
    expert_names: list[str],
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    all_norms = [value for row in rows for value in row["metric_norms"]]
    if not all_norms or min(all_norms) <= 0:
        raise ValueError("all section norms must be positive for the raw figure")
    norm_limits = (min(all_norms) * 0.75, max(all_norms) * 1.35)
    pairs = tuple(combinations(range(len(expert_names)), 2))
    produced = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
        }
    )
    for node in NODE_ORDER:
        node_rows = sorted(
            (row for row in rows if row["node"] == node),
            key=lambda row: row["layer"],
        )
        layers = [row["layer"] + 1 for row in node_rows]
        pair_values = {
            pair: [row["rho"][pair[0]][pair[1]] for row in node_rows]
            for pair in pairs
        }
        cosine_limit = symmetric_limit(
            [value for values in pair_values.values() for value in values]
        )
        fig, (ax_cos, ax_norm) = plt.subplots(
            2,
            1,
            figsize=(7.1, 4.8),
            sharex=True,
            gridspec_kw={"height_ratios": (1.05, 1.0), "hspace": 0.18},
        )
        for pair, color in zip(pairs, PAIR_COLORS):
            ax_cos.plot(
                layers,
                pair_values[pair],
                color=color,
                marker="o",
                markersize=3.2,
                linewidth=1.55,
                label=f"{expert_names[pair[0]]} - {expert_names[pair[1]]}",
            )
        ax_cos.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
        ax_cos.set_ylim(-cosine_limit, cosine_limit)
        ax_cos.set_ylabel(r"Cosine overlap $\rho_{ij}$")
        ax_cos.set_title("(a) Signed directional overlap", loc="left", fontsize=10.5, pad=7)
        ax_cos.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax_cos.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(3, len(pairs)),
            frameon=False,
            handlelength=2.4,
            columnspacing=1.4,
        )
        for index, (expert, color) in enumerate(zip(expert_names, EXPERT_COLORS)):
            ax_norm.plot(
                layers,
                [row["metric_norms"][index] for row in node_rows],
                color=color,
                marker="o",
                markersize=3.2,
                linewidth=1.55,
                label=expert,
            )
        ax_norm.set_yscale("log")
        ax_norm.set_ylim(*norm_limits)
        ax_norm.set_ylabel(r"Section norm $n_i$")
        ax_norm.set_xlabel("Transformer layer")
        ax_norm.set_title("(b) Gram-section magnitude", loc="left", fontsize=10.5, pad=7)
        ax_norm.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(3, len(expert_names)),
            frameon=False,
            handlelength=2.4,
            columnspacing=1.8,
        )
        final_layer = max(layers)
        ticks = sorted({1, final_layer, *range(4, final_layer + 1, 4)})
        ax_norm.set_xticks(ticks)
        ax_norm.set_xlim(0.5, final_layer + 0.5)
        for axis in (ax_cos, ax_norm):
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.75)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(direction="out", length=3.5, width=0.8)
        path = output_dir / f"{node}_raw_geometry.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        produced.append(path)
        print(f"[stage2] wrote {path}", flush=True)
    return produced


def format_matrix_value(value: float) -> str:
    if abs(value) < 0.005:
        return "0"
    return f"{value:.2f}"


def render_normbalanced_figures(
    rows: list[dict],
    expert_names: list[str],
    output_dir: Path,
) -> list[Path]:
    import io

    import matplotlib
    from PIL import Image

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def panel(matrix: np.ndarray, title: str) -> Image.Image:
        fig, ax = plt.subplots(figsize=(1.55, 1.55), dpi=170)
        image = ax.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks(range(len(expert_names)))
        ax.set_yticks(range(len(expert_names)))
        ax.set_xticklabels(expert_names, rotation=45, ha="right", fontsize=5.7)
        ax.set_yticklabels(expert_names, fontsize=5.7)
        ax.tick_params(length=0, pad=1)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                ax.text(
                    col_index,
                    row_index,
                    format_matrix_value(float(matrix[row_index, col_index])),
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="black",
                )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        colorbar.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.35)
            spine.set_color("0.55")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches=None, pad_inches=0)
        plt.close(fig)
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
        result.load()
        buffer.close()
        return result

    def stitch(images: list[Image.Image], path: Path) -> None:
        panel_width = max(image.width for image in images)
        panel_height = max(image.height for image in images)
        columns = 6
        pad_x, pad_y = 18, 20
        rows_count = math.ceil(len(images) / columns)
        canvas = Image.new(
            "RGB",
            (
                columns * panel_width + (columns - 1) * pad_x,
                rows_count * panel_height + (rows_count - 1) * pad_y,
            ),
            (255, 255, 255),
        )
        for index, image in enumerate(images):
            row_index, col_index = divmod(index, columns)
            canvas.paste(
                image,
                (
                    col_index * (panel_width + pad_x),
                    row_index * (panel_height + pad_y),
                ),
            )
        canvas.save(path)
        canvas.close()
        for image in images:
            image.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    produced = []
    for node in NODE_ORDER:
        node_rows = sorted(
            (row for row in rows if row["node"] == node),
            key=lambda row: row["layer"],
        )
        images = [
            panel(
                np.asarray(row["normbalanced_gamma"], dtype=np.float64),
                f"Layer {row['layer'] + 1}",
            )
            for row in node_rows
        ]
        path = output_dir / f"{node}_normbalanced_mosaic.png"
        stitch(images, path)
        produced.append(path)
        print(f"[stage2] wrote {path} panels={len(images)}", flush=True)
    return produced


def write_outputs(
    config_path: Path,
    config: dict,
    stage1_dir: Path,
    out_dir: Path,
    rows: list[dict],
    norm_eps: float,
    base: str,
    expert_paths: list[str],
    render_figures: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    d_values = np.asarray([row["D_v"] for row in rows], dtype=np.float64)
    count_bad = int(np.sum(d_values > 1.0))
    decision = "NormBalanced-RGM" if count_bad else "Raw RGM"
    expert_names = [str(item["name"]) for item in config["experts"]]
    summary = {
        "stage": "stage2_gate",
        "setting": config["name"],
        "setting_id": config["id"],
        "config": str(config_path),
        "stage1_dir": str(stage1_dir),
        "base": base,
        "experts": expert_names,
        "expert_paths": expert_paths,
        "layers": config["layers"],
        "nodes": list(NODE_ORDER),
        "norm_eps": norm_eps,
        "D_v_p95": float(np.quantile(d_values, 0.95)),
        "D_v_max": float(np.max(d_values)),
        "D_v_gt_1_nodes": count_bad,
        "num_nodes": len(rows),
        "decision": decision,
        "branch": "normbalanced" if decision == "NormBalanced-RGM" else "raw",
    }

    raw_gamma = np.asarray([row["raw_gamma"] for row in rows], dtype=np.float64)
    normbalanced_gamma = np.asarray(
        [row["normbalanced_gamma"] for row in rows], dtype=np.float64
    )
    metric_norms = np.asarray([row["metric_norms"] for row in rows], dtype=np.float64)
    rho = np.asarray([row["rho"] for row in rows], dtype=np.float64)
    np.savez_compressed(
        out_dir / "node_statistics.npz",
        raw_gamma=raw_gamma,
        normbalanced_gamma=normbalanced_gamma,
        metric_norms=metric_norms,
        rho=rho,
        D_v=np.asarray([row["D_v"] for row in rows], dtype=np.float64),
        D_by_expert=np.asarray([row["D_by_expert"] for row in rows], dtype=np.float64),
        layer=np.asarray([row["layer"] for row in rows], dtype=np.int64),
        node=np.asarray([row["node"] for row in rows], dtype=object),
        label=np.asarray([row["label"] for row in rows], dtype=object),
        experts=np.asarray(expert_names, dtype=object),
    )
    with (out_dir / "node_statistics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    fields = [
        "setting",
        "setting_id",
        "layer",
        "node",
        "label",
        "D_v",
        "D_by_expert",
        "metric_norms",
        "rho",
        "raw_gamma",
        "normbalanced_gamma",
    ]
    with (out_dir / "decision_node_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "setting": config["name"],
                    "setting_id": config["id"],
                    **{
                        key: json.dumps(row[key]) if isinstance(row[key], list) else row[key]
                        for key in fields[2:]
                    },
                }
            )

    table_fields = ["Setting", "D_v p95", "D_v max", "D_v > 1 nodes", "Decision"]
    with (out_dir / "decision_table_row.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields)
        writer.writeheader()
        writer.writerow(
            {
                "Setting": config["name"],
                "D_v p95": f"{summary['D_v_p95']:.4f}",
                "D_v max": f"{summary['D_v_max']:.4f}",
                "D_v > 1 nodes": f"{count_bad}/{len(rows)}",
                "Decision": decision,
            }
        )

    figures_dir = out_dir / "figures"
    if render_figures:
        if figures_dir.exists():
            for path in figures_dir.glob("*.png"):
                path.unlink()
        if summary["branch"] == "raw":
            produced = render_raw_figures(rows, expert_names, figures_dir)
        else:
            produced = render_normbalanced_figures(rows, expert_names, figures_dir)
        summary["figures"] = [str(path) for path in produced]
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: compute pre-merge cross-pressure statistics and gate."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage1-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = load_setting(args.config, validate_paths=True)
    base, expert_paths, expert_names, layers = model_paths(config)
    stage1_dir = args.stage1_dir.expanduser().resolve()
    stage1_summary_path = stage1_dir / "summary.json"
    if not stage1_summary_path.is_file():
        raise FileNotFoundError(f"missing Stage 1 summary: {stage1_summary_path}")
    stage1_summary = json.loads(stage1_summary_path.read_text(encoding="utf-8"))
    if stage1_summary.get("experts") != expert_names:
        raise ValueError("Stage 1 expert order does not match the selected setting")
    if stage1_summary.get("layers") != layers:
        raise ValueError("Stage 1 layer order does not match the selected setting")
    norm_eps = float(defaults(config)["norm_eps"])
    rows = compute_rows(stage1_dir, expert_names, layers, norm_eps)
    write_outputs(
        args.config.expanduser().resolve(),
        config,
        stage1_dir,
        args.out_dir.expanduser().resolve(),
        rows,
        norm_eps,
        base,
        expert_paths,
        render_figures=not args.no_figures,
    )


if __name__ == "__main__":
    main()
