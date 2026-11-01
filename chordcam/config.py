"""YAML config loading."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config; relative `paths` entries resolve against the repo root."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: str(REPO_ROOT / v) if not Path(v).is_absolute() else v
                    for k, v in cfg.get("paths", {}).items()}
    return cfg
