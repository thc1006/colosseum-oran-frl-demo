# 檔案：src/colosseum_oran_frl_demo/agents/fed_server.py
"""
簡易 FedAvg – 使用 torch.stack 保留 dtype，避免隱式變成 float64。
"""
from __future__ import annotations
import torch, copy
from typing import List, Dict

@torch.no_grad()
def fedavg(client_model_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    接收一個模型狀態字典的列表，回傳一個平均後的狀態字典。
    """
    if not client_model_states:
        return {}
    
    # 取得第一個客戶端的模型鍵
    keys = client_model_states[0].keys()
    
    # 計算每個參數的平均值
    avg_state = {
        k: torch.mean(
            torch.stack([s[k].float() for s in client_model_states]), dim=0
        )
        for k in keys
    }
    
    return avg_state
