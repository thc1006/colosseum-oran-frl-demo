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

    def __post_init__(self):
        if not self.LR > 0:
            raise ValueError("Learning rate (LR) must be positive.")
        if not 0 <= self.GAMMA <= 1:
            raise ValueError("Gamma (GAMMA) must be between 0 and 1.")
        if not 0 < self.EPS_DECAY <= 1:
            raise ValueError("Epsilon decay (EPS_DECAY) must be between 0 and 1.")
        if not self.LOCAL_STEPS > 0:
            raise ValueError("Local steps (LOCAL_STEPS) must be positive.")


def hp_dict() -> Dict[str, Any]:
    """SAFELY turn HP into JSON-serialisable dict"""
    return asdict(HP())
