# ─── src/colosseum_oran_frl_demo/config.py ───
from dataclasses import asdict, dataclass
from pathlib import Path

_ROOT = Path.cwd()


@dataclass
class Paths:
    """Dataclass to store and manage file paths."""

    root: Path = _ROOT
    raw_data: Path = (
        _ROOT / """src""" / """colosseum_oran_frl_demo""" / """data""" / """raw"""
    )
    processed: Path = (
        _ROOT / """src""" / """colosseum_oran_frl_demo""" / """data""" / """processed"""
    )
    outputs: Path = _ROOT / """outputs"""


@dataclass
class HP:
    """Dataclass to store and manage hyperparameters."""

    lr: float = 1e-3
    gamma: float = 0.95
    eps_decay: float = 0.998
    local_steps: int = 2_000


def hp_dict() -> dict:
    """SAFELY turn HP into JSON-serialisable dict"""
    return asdict(HP())
