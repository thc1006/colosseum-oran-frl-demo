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
