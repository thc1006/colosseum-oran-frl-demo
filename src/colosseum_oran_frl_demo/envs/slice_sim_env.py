# 檔案：src/colosseum_oran_frl_demo/envs/slice_sim_env.py
"""
A lightweight, Gym-style environment that re-plays KPI traces.
The original Notebook class `SliceSimEnv` is copied verbatim
(and trimmed for standalone use).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class SliceSimEnv:
    """
    Replays KPI rows for a single gNB / selected slices.

    * observation = [tput_slice0, lat_slice0, tput_slice2, lat_slice2]
    * action      = {0: prioritise eMBB, 1: balanced, 2: prioritise URLLC}
    * reward      = weighted combo (see _compute_reward)
    """

    def __init__(self, df_data: pd.DataFrame, gnb_id: int = 1, slice_ids=(0, 2)):
        self.df_data = df_data[df_data["BS_ID"] == gnb_id].copy()
        self.slices = slice_ids
        self.ts_col = "timestamp"
        self._prep()
        self.reset()

    # ---------- helpers --------------------------------------------------- #
    def _prep(self):
        self.df_data.sort_values(self.ts_col, inplace=True)
        self.ts = self.df_data[self.ts_col].unique()
        self.cursor = 0

    def _state_from_row(self, sub: pd.DataFrame) -> np.ndarray:
        out = []
        for sid in self.slices:
            row = sub[sub["Slice_ID"] == sid]
            if row.empty:
                out.extend([0.0, 0.0])
            else:
                out.append(row["Throughput_DL_Mbps"].iloc[0])
                out.append(row["Latency_proxy_ms"].iloc[0])
        return np.asarray(out, dtype=np.float32)

    def _compute_reward(self, state: np.ndarray) -> float:
        tput0, lat0, tput2, lat2 = state
        # simple linear combo; tweak weights as you wish
        return 0.3 * tput0 - 0.7 * lat2

    # ---------- Gym-like API --------------------------------------------- #
    def reset(self):
        self.cursor = 0
        sub = self.df_data[self.df_data[self.ts_col] == self.ts[self.cursor]]
        return self._state_from_row(sub), {}

    def step(self, action: int):
        self.cursor = (self.cursor + 1) % len(self.ts)
        sub = self.df_data[self.df_data[self.ts_col] == self.ts[self.cursor]]
        state = self._state_from_row(sub)
        reward = self._compute_reward(state)
        terminated = False
        truncated = False
        info = {"timestamp": self.ts[self.cursor]}
        return state, reward, terminated, truncated, info

    @property
    def action_size(self):
        return 3

    @property
    def state_size(self):
        return 4
