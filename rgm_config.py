#!/usr/bin/env python3
"""Configuration helpers for the three RGM paper settings."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


SUPPORTED_SETTINGS: dict[str, dict[str, Any]] = {
    "qwen25_7b_instruct": {
        "expert_names": ("CURE", "ToolRL", "MemAgent"),
        "expected_branch": "raw",
    },
    "deepseek_r1_distill_qwen_1p5b": {
        "expert_names": (
            "Archer2.0-Code-1.5B-Preview",
            "JustRL-DeepSeek-1.5B",
        ),
        "expected_branch": "normbalanced",
    },
    "openmath_nemotron_1p5b": {
        "expert_names": (
            "QuestA-Nemotron-1.5B",
            "JustRL-Nemotron-1.5B",
        ),
        "expected_branch": "normbalanced",
    },
}
EXPECTED_LAYERS = tuple(range(28))
VALID_DTYPES = {"bfloat16", "float16", "float32"}
_UNRESOLVED_PATH = re.compile(r"PATH_TO_|<[^>]+>|\$\{?\w+\}?")


def resolve_path(value: str, config_path: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def is_path_placeholder(value: str) -> bool:
    return bool(_UNRESOLVED_PATH.search(value))


def require_checkpoint(value: str, label: str, config_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{config_path}: {label} path is empty")
    if is_path_placeholder(value):
        raise ValueError(
            f"{config_path}: replace the unresolved path for {label}: {value}"
        )

    path = resolve_path(value, config_path)
    if not path.is_dir():
        raise FileNotFoundError(f"{config_path}: {label} directory not found: {path}")
    has_single = (path / "model.safetensors").is_file()
    has_index = (path / "model.safetensors.index.json").is_file()
    if not (has_single or has_index):
        raise FileNotFoundError(
            f"{config_path}: {label} has no model.safetensors or "
            f"model.safetensors.index.json: {path}"
        )
    return path


def validate_setting(config: dict[str, Any], config_path: Path) -> None:
    for key in ("id", "name", "base", "experts", "layers"):
        if key not in config:
            raise ValueError(f"{config_path}: missing required key {key!r}")

    setting_id = config["id"]
    if setting_id not in SUPPORTED_SETTINGS:
        allowed = ", ".join(sorted(SUPPORTED_SETTINGS))
        raise ValueError(
            f"{config_path}: unsupported setting id {setting_id!r}; expected one of: {allowed}"
        )
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError(f"{config_path}: name must be a non-empty string")
    if not isinstance(config["base"], str) or not config["base"].strip():
        raise ValueError(f"{config_path}: base must be a non-empty path string")

    experts = config["experts"]
    if not isinstance(experts, list) or not experts:
        raise ValueError(f"{config_path}: experts must be a non-empty list")
    for index, expert in enumerate(experts):
        if not isinstance(expert, dict) or set(expert) != {"name", "path"}:
            raise ValueError(
                f"{config_path}: experts[{index}] must contain exactly name and path"
            )
        if not isinstance(expert["name"], str) or not expert["name"].strip():
            raise ValueError(f"{config_path}: experts[{index}].name is invalid")
        if not isinstance(expert["path"], str) or not expert["path"].strip():
            raise ValueError(f"{config_path}: experts[{index}].path is invalid")

    expert_names = tuple(expert["name"] for expert in experts)
    expected_names = SUPPORTED_SETTINGS[setting_id]["expert_names"]
    if expert_names != expected_names:
        raise ValueError(
            f"{config_path}: setting {setting_id!r} requires experts in order "
            f"{expected_names}, found {expert_names}"
        )
    if len(set(expert_names)) != len(expert_names):
        raise ValueError(f"{config_path}: expert names must be unique")

    layers = config["layers"]
    if not isinstance(layers, list) or any(
        isinstance(layer, bool) or not isinstance(layer, int) for layer in layers
    ):
        raise ValueError(f"{config_path}: layers must be a list of integers")
    if tuple(layers) != EXPECTED_LAYERS:
        raise ValueError(f"{config_path}: paper settings require layers 0..27 in order")

    unknown_top_level = set(config) - {
        "id",
        "name",
        "base",
        "experts",
        "layers",
        "defaults",
    }
    if unknown_top_level:
        raise ValueError(
            f"{config_path}: unknown top-level keys: {sorted(unknown_top_level)}"
        )
    configured_defaults = config.get("defaults", {})
    if not isinstance(configured_defaults, dict):
        raise ValueError(f"{config_path}: defaults must be an object")
    unknown_defaults = set(configured_defaults) - {"ridge", "norm_eps", "out_dtype"}
    if unknown_defaults:
        raise ValueError(f"{config_path}: unknown defaults: {sorted(unknown_defaults)}")

    values = defaults(config)
    if not isinstance(values["ridge"], (int, float)) or values["ridge"] < 0:
        raise ValueError(f"{config_path}: ridge must be a non-negative number")
    if not isinstance(values["norm_eps"], (int, float)) or values["norm_eps"] <= 0:
        raise ValueError(f"{config_path}: norm_eps must be a positive number")
    if values["out_dtype"] not in VALID_DTYPES:
        raise ValueError(
            f"{config_path}: out_dtype must be one of {sorted(VALID_DTYPES)}"
        )


def load_setting(path: str | Path, validate_paths: bool = True) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"setting config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    validate_setting(config, config_path)
    config["_config_path"] = str(config_path)
    if validate_paths:
        require_checkpoint(config["base"], "base", config_path)
        for expert in config["experts"]:
            require_checkpoint(expert["path"], f"expert {expert['name']}", config_path)
    return config


def model_paths(config: dict[str, Any]) -> tuple[str, list[str], list[str], list[int]]:
    config_path = Path(config["_config_path"])
    return (
        str(require_checkpoint(config["base"], "base", config_path)),
        [
            str(require_checkpoint(item["path"], f"expert {item['name']}", config_path))
            for item in config["experts"]
        ],
        [str(item["name"]) for item in config["experts"]],
        [int(layer) for layer in config["layers"]],
    )


def defaults(config: dict[str, Any]) -> dict[str, Any]:
    values = {
        "ridge": 1e-3,
        "norm_eps": 1e-12,
        "out_dtype": "bfloat16",
    }
    values.update(config.get("defaults", {}))
    return values


def setting_spec(config: dict[str, Any]) -> dict[str, Any]:
    return SUPPORTED_SETTINGS[str(config["id"])]
