import pytest
from pathlib import Path
from colosseum_oran_frl_demo.config import Paths, HP, hp_dict


def test_paths_exist():
    assert Paths.ROOT.is_dir()
    assert Paths.RAW_DATA.is_dir()
    assert Paths.PROCESSED.is_dir()
    assert Paths.OUTPUTS.is_dir()

def test_hyperparameters_types():
    assert isinstance(HP.LR, float)
    assert isinstance(HP.GAMMA, float)
    assert isinstance(HP.EPS_DECAY, float)
    assert isinstance(HP.LOCAL_STEPS, int)

def test_hp_dict_content():
    d = hp_dict()
    assert isinstance(d, dict)
    assert d["LR"] == HP.LR
    assert d["GAMMA"] == HP.GAMMA
    assert d["EPS_DECAY"] == HP.EPS_DECAY
    assert d["LOCAL_STEPS"] == HP.LOCAL_STEPS

def test_hyperparameters_invalid_values():
    with pytest.raises(ValueError, match=r"Learning rate \(LR\) must be positive."):
        HP(LR=-0.1)

    with pytest.raises(ValueError, match=r"Gamma \(GAMMA\) must be between 0 and 1."):
        HP(GAMMA=1.5)

    with pytest.raises(ValueError, match=r"Epsilon decay \(EPS_DECAY\) must be between 0 and 1."):
        HP(EPS_DECAY=1.1)

    with pytest.raises(ValueError, match=r"Local steps \(LOCAL_STEPS\) must be positive."):
        HP(LOCAL_STEPS=0)
