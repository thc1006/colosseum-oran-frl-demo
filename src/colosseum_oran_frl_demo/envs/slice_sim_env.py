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

    def __init__(
        self, df_data: pd.DataFrame, gnb_id: int = 1, slice_ids: Tuple[int, int] = (0, 2)
    ):
        """
        Initializes the SliceSimEnv.

        Args:
            df_data: The input DataFrame containing KPI traces.
            gnb_id: The gNB ID to filter data for.
            slice_ids: A tuple of slice IDs to consider for observations.
        """
        self.df_data = df_data[df_data["BS_ID"] == gnb_id].copy()
        self.slices = slice_ids
        self.ts_col = "timestamp"
        self._prep()
        if not self.ts.any():
            raise ValueError(f"No data found for gNB ID {gnb_id} in the provided DataFrame.")
        self.reset()

    # ---------- helpers --------------------------------------------------- #
    def _prep(self) -> None:
        """
        Prepares the DataFrame by sorting values and extracting unique timestamps.
        """
        self.df_data.sort_values(self.ts_col, inplace=True)
        self.ts = self.df_data[self.ts_col].unique()
        self.cursor = 0

    def _state_from_row(self, sub: pd.DataFrame) -> np.ndarray:
        """
        Extracts the state from a sub-DataFrame row.

        Args:
            sub: A sub-DataFrame containing KPI data for a specific timestamp.

        Returns:
            A NumPy array representing the state.
        """
        out = []
        for sid in self.slices:
            row = sub[sub["Slice_ID"] == sid]
            if row.empty:
                out.extend([0.0, 0.0])
            else:
                out.append(row["tx_brate downlink [Mbps]"].iloc[0])
                out.append(row["dl_buffer [bytes]"].iloc[0])
        return np.asarray(out, dtype=np.float32)

    def _compute_reward(self, state: np.ndarray) -> float:
        """
        Computes the reward based on the current state.

        Args:
            state: The current state as a NumPy array.

        Returns:
            The calculated reward.
        """
        tput0, lat0, tput2, lat2 = state
        # simple linear combo; tweak weights as you wish
        return 0.3 * tput0 - 0.7 * lat2

    # ---------- Gym-like API --------------------------------------------- #
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to its initial state.

        Returns:
            A tuple containing the initial state (NumPy array) and an empty dictionary.
        """
        self.cursor = 0
        sub = self.df_data[self.df_data[self.ts_col] == self.ts[self.cursor]]
        return self._state_from_row(sub), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Performs a step in the environment given an action.

        Args:
            action: The action to take.

        Returns:
            A tuple containing:
                - next_state (np.ndarray): The new state after the action.
                - reward (float): The reward received.
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode has been truncated.
                - info (Dict): Additional information, including the timestamp.
        """
        self.cursor = (self.cursor + 1) % len(self.ts)
        sub = self.df_data[self.df_data[self.ts_col] == self.ts[self.cursor]]
        state = self._state_from_row(sub)
        reward = self._compute_reward(state)
        terminated = False
        truncated = False
        info = {"timestamp": self.ts[self.cursor]}
        return state, reward, terminated, truncated, info

    @property
    def action_size(self) -> int:
        """
        Returns the size of the action space.

        Returns:
            The number of possible actions.
        """
        return 3

    @property
    def state_size(self) -> int:
        """
        Returns the size of the state space.

        Returns:
            The number of state features.
        """
        return 4
