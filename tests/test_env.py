# 檔案: tests/test_env.py

import numpy as np
import pandas as pd
from colosseum_oran_frl_demo.envs.slice_sim_env import SliceSimEnv

def test_env_step():
    df = pd.DataFrame({
        "timestamp":[0,0,1,1],
        "BS_ID":[1,1,1,1],
        "Slice_ID":[0,2,0,2],
        "Throughput_DL_Mbps":[10,5,12,6],
        "Latency_proxy_ms":[15,8,14,7]
    })
    env = SliceSimEnv(df, gnb_id=1)
    s,_ = env.reset()
    assert s.dtype == np.float32
    s2,r,_,_,info = env.step(1)

    # 修正：允許獎勵 r 的型別為 float 或任何 numpy 的浮點數型別
    assert len(s2) == 4 and isinstance(r, (float, np.floating))
