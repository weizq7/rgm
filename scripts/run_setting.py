#!/usr/bin/env python3
"""Run the three RGM stages for one paper setting."""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rgm_config import defaults, load_setting, model_paths


STAGE_ORDER = ("gram", "gate", "merge")


def printable_command(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def run_command(command: list[str], dry_run: bool) -> None:
    print(f"[command] {printable_command(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_stages(values: list[str]) -> list[str]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("--stages all cannot be combined with individual stages")
        return list(STAGE_ORDER)
    selected = set(values)
    return [stage for stage in STAGE_ORDER if stage in selected]


def files_exist(paths: Sequence[Path]) -> bool:
    return bool(paths) and all(path.is_file() for path in paths)


def stage1_outputs(stage_dir: Path, layers: list[int]) -> list[Path]:
    return [
        stage_dir / "summary.json",
        *(stage_dir / "sections" / f"layer_{layer:02d}.safetensors" for layer in layers),
    ]


def stage2_outputs(
    stage_dir: Path,
    branch: str | None,
    include_figures: bool = True,
) -> list[Path]:
    outputs = [
        stage_dir / "summary.json",
        stage_dir / "node_statistics.npz",
        stage_dir / "node_statistics.jsonl",
        stage_dir / "decision_node_rows.csv",
        stage_dir / "decision_table_row.csv",
    ]
    if include_figures and branch == "raw":
        outputs.extend(
            stage_dir / "figures" / f"{node}_raw_geometry.png"
            for node in ("attn_in", "post_attn", "post_mlp")
        )
    elif include_figures and branch == "normbalanced":
        outputs.extend(
            stage_dir / "figures" / f"{node}_normbalanced_mosaic.png"
            for node in ("attn_in", "post_attn", "post_mlp")
        )
    return outputs


def gate_branch(stage2_dir: Path) -> str | None:
    summary_path = stage2_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    branch = summary.get("branch")
    return branch if branch in {"raw", "normbalanced"} else None


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def retain_public_outputs(setting_dir: Path) -> None:
    """Keep only the figures and merged checkpoint in a completed run."""
    retained = {"stage2_gate", "stage3_merge"}
    for child in setting_dir.iterdir():
        if child.name not in retained:
            remove_path(child)

    stage2_dir = setting_dir / "stage2_gate"
    if stage2_dir.is_dir():
        for child in stage2_dir.iterdir():
            if child.name != "figures":
                remove_path(child)

    stage3_dir = setting_dir / "stage3_merge"
    if stage3_dir.is_dir():
        for child in stage3_dir.iterdir():
            if child.name != "merged_model":
                remove_path(child)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RGM: state sections, pre-merge gate, then coefficient merge."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--branch",
        choices=("auto", "raw", "normbalanced"),
        default="auto",
        help="Use the Stage 2 gate or explicitly select a Stage 3 branch.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all", *STAGE_ORDER),
        default=["all"],
        help="Stages to execute in canonical order: gram, gate, merge.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected stages even when their outputs exist.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip the three branch-specific figures in Stage 2.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep Stage 1/2/3 intermediate files after a complete run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config schema and print stage commands.",
    )
    args = parser.parse_args()

    config = load_setting(args.config, validate_paths=not args.dry_run)
    values = defaults(config)
    config_path = Path(config["_config_path"])
    if args.dry_run:
        base = str(config["base"])
        experts = [str(item["path"]) for item in config["experts"]]
        expert_names = [str(item["name"]) for item in config["experts"]]
        layers = [int(layer) for layer in config["layers"]]
    else:
        base, experts, expert_names, layers = model_paths(config)

    stages = parse_stages(args.stages)
    output_root = args.output_root.expanduser().resolve()
    setting_dir = output_root / str(config["id"])
    stage1_dir = setting_dir / "stage1_gram"
    stage2_dir = setting_dir / "stage2_gate"
    stage3_dir = setting_dir / "stage3_merge"
    python = sys.executable

    if not args.dry_run:
        setting_dir.mkdir(parents=True, exist_ok=True)

    if "gram" in stages:
        outputs = stage1_outputs(stage1_dir, layers)
        if not args.force and files_exist(outputs):
            print(f"[skip] gram outputs already exist in {stage1_dir}", flush=True)
        else:
            run_command(
                [
                    python,
                    str(REPO_ROOT / "rgm_stage1_gram.py"),
                    "--config",
                    str(config_path),
                    "--out-dir",
                    str(stage1_dir),
                    "--device",
                    args.device,
                ],
                args.dry_run,
            )

    if "gate" in stages:
        current_branch = gate_branch(stage2_dir)
        outputs = stage2_outputs(
            stage2_dir,
            current_branch,
            include_figures=not args.no_figures,
        )
        if not args.force and current_branch and files_exist(outputs):
            print(f"[skip] gate outputs already exist in {stage2_dir}", flush=True)
        else:
            command = [
                python,
                str(REPO_ROOT / "rgm_stage2_gate.py"),
                "--config",
                str(config_path),
                "--stage1-dir",
                str(stage1_dir),
                "--out-dir",
                str(stage2_dir),
            ]
            if args.no_figures:
                command.append("--no-figures")
            run_command(command, args.dry_run)

    if "merge" in stages:
        current_branch = args.branch
        if current_branch == "auto" and not args.dry_run:
            current_branch = gate_branch(stage2_dir)
            if current_branch is None:
                raise FileNotFoundError(
                    "--branch auto requires stage2_gate/summary.json; "
                    "run the gate stage or select a branch explicitly"
                )
        coeff_summary = stage3_dir / "coefficients" / "rgm_coeff_summary.json"
        merged_model = stage3_dir / "merged_model"
        merge_outputs = (
            stage3_dir / "summary.json",
            coeff_summary,
            merged_model / "metric_sheaf_edge_coefficients.json",
            merged_model / "fallback_tensors.json",
        )
        if not args.force and files_exist(merge_outputs):
            print(f"[skip] merge outputs already exist in {stage3_dir}", flush=True)
        else:
            run_command(
                [
                    python,
                    str(REPO_ROOT / "rgm_stage3_merge.py"),
                    "--config",
                    str(config_path),
                    "--stage2-dir",
                    str(stage2_dir),
                    "--out-dir",
                    str(stage3_dir),
                    "--branch",
                    current_branch,
                    "--ridge",
                    str(values["ridge"]),
                    "--out-dtype",
                    str(values["out_dtype"]),
                ],
                args.dry_run,
            )

    if not args.dry_run and args.keep_intermediates:
        manifest = {
            "setting_id": config["id"],
            "setting_name": config["name"],
            "config": str(config_path),
            "base": base,
            "experts": [
                {"name": name, "path": path}
                for name, path in zip(expert_names, experts)
            ],
            "layers": layers,
            "ridge": values["ridge"],
            "norm_eps": values["norm_eps"],
            "out_dtype": values["out_dtype"],
            "requested_branch": args.branch,
            "resolved_branch": gate_branch(stage2_dir),
            "stages": stages,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (setting_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    elif not args.dry_run and set(stages) == set(STAGE_ORDER):
        retain_public_outputs(setting_dir)
    print(f"[DONE] setting={config['id']} output={setting_dir}", flush=True)


if __name__ == "__main__":
    main()
