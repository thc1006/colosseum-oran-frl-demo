# 檔案: tests/test_env.py

import numpy as np
import pandas as pd
import pytest
from colosseum_oran_frl_demo.envs.slice_sim_env import SliceSimEnv


def test_env_step() -> None:
    """
    Tests the step method of SliceSimEnv.
    """
    df = pd.DataFrame(
        {
            "timestamp": [0, 0, 1, 1],
            "BS_ID": [1, 1, 1, 1],
            "Slice_ID": [0, 2, 0, 2],
            "Throughput_DL_Mbps": [10, 5, 12, 6],
            "Latency_proxy_ms": [15, 8, 14, 7],
        }
    )
    env = SliceSimEnv(df, gnb_id=1)
    s, _ = env.reset()
    assert s.dtype == np.float32
    s2, r, terminated, truncated, info = env.step(1)

    assert len(s2) == 4
    assert isinstance(r, (float, np.floating))
    assert not terminated
    assert not truncated
    assert "timestamp" in info

def test_env_reset_no_data() -> None:
    """
    Tests that SliceSimEnv raises ValueError when no data is found for gnb_id.
    """
    df = pd.DataFrame(
        {
            "timestamp": [0, 0, 1, 1],
            "BS_ID": [99, 99, 99, 99],  # No data for gnb_id=1
            "Slice_ID": [0, 2, 0, 2],
            "Throughput_DL_Mbps": [10, 5, 12, 6],
            "Latency_proxy_ms": [15, 8, 14, 7],
        }
    )
    with pytest.raises(ValueError, match="No data found for gNB ID 1 in the provided DataFrame."):
        SliceSimEnv(df, gnb_id=1)
