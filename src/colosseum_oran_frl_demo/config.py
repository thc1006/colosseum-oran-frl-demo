# ─── src/colosseum_oran_frl_demo/config.py ───
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any

_ROOT: Path = Path.cwd()


@dataclass
class Paths:
    """Defines paths for data and outputs."""
    ROOT: Path = _ROOT
    RAW_DATA: Path = _ROOT / "src" / "colosseum_oran_frl_demo" / "data" / "raw"
    PROCESSED: Path = _ROOT / "src" / "colosseum_oran_frl_demo" / "data" / "processed"
    OUTPUTS: Path = _ROOT / "outputs"


@dataclass
class HP:
    """Defines hyperparameters for the reinforcement learning agent."""
    LR: float = 1e-3
    GAMMA: float = 0.95
    EPS_DECAY: float = 0.998
    LOCAL_STEPS: int = 2_000


def hp_dict() -> Dict[str, Any]:
    """SAFELY turn HP into JSON-serialisable dict"""
    return asdict(HP())
