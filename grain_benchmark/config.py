"""The two representation regimes and their method-specific settings."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("grain_only", "grain_volcashdb")
METHODS = ("diva", "deepdpm", "ddpm", "kmeans", "byol", "propos", "autopropos")


def load_config(path=None, regime=None):
    if path is None:
        if regime not in REGIMES:
            raise ValueError("Pass --config or --regime grain_only/grain_volcashdb")
        path = ROOT / "configs" / f"{regime}.yaml"
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("regime") not in REGIMES:
        raise ValueError(f"Invalid benchmark configuration: {path}")
    if regime is not None and regime != config["regime"]:
        raise ValueError("--regime conflicts with the selected configuration")
    expected = {"regime", "image", *METHODS}
    if set(config) != expected:
        raise ValueError(f"Unsupported/missing configuration sections: {set(config) ^ expected}")
    for section in expected - {"regime"}:
        if not isinstance(config[section], dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping")
    return config


def method_settings(method, path=None, regime=None):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    config = load_config(path, regime)
    settings = deepcopy(config.get("image", {})) if method in ("byol", "propos", "autopropos") else {}
    settings.update(deepcopy(config[method]))
    return config["regime"], settings
