# ─── src/colosseum_oran_frl_demo/agents/fed_server.py ───
"""
簡易 FedAvg – 使用 torch.stack 保留 dtype，避免隱式變成 float64。
"""
from __future__ import annotations
import torch, copy
from typing import List
from colosseum_oran_frl_demo.agents.rl_agent import RLAgent

@torch.no_grad()
def fedavg(clients: List[RLAgent]) -> None:
    ks = clients[0].model.state_dict().keys()
    avg_state = {k: torch.mean(torch.stack([c.model.state_dict()[k] for c in clients]), dim=0)
                 for k in ks}
    for c in clients:
        c.model.load_state_dict(avg_state)
