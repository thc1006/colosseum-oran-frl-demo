# 檔案：src/colosseum_oran_frl_demo/agents/fed_server.py
"""
簡易 FedAvg – 使用 torch.stack 保留 dtype，避免隱式變成 float64。
"""
from __future__ import annotations
import torch
from typing import List, Dict


@torch.no_grad()
def fedavg(
    client_model_states: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Aggregates client model states using the Federated Averaging (FedAvg) algorithm.

    Args:
        client_model_states: A list of dictionaries, where each dictionary represents
                             the state_dict of a client's model.

    Returns:
        A dictionary representing the averaged global model state.
    """
    if not client_model_states:
        return {}

    # Assumption: All client models have identical architectures and parameter names.
    # If different architectures are to be supported, this function would require
    # significant modifications (e.g., handling parameter mismatches, interpolation).
    keys = client_model_states[0].keys()

    # Calculate the average for each parameter
    avg_state = {
        k: torch.mean(torch.stack([s[k].float() for s in client_model_states]), dim=0)
        for k in keys
    }

    return avg_state
