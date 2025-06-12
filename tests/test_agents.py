# 檔案: tests/test_agents.py

import numpy as np
import torch # 修正：匯入 torch
from colosseum_oran_frl_demo.agents.rl_agent import RLAgent
from colosseum_oran_frl_demo.agents.fed_server import fedavg

def test_agent_and_fedavg():
    a1 = RLAgent(4, 3)
    a2 = RLAgent(4, 3)
    
    # 讓兩個模型的初始權重不同
    for p1, p2 in zip(a1.model.parameters(), a2.model.parameters()):
        p2.data = torch.randn_like(p1.data)
        assert not torch.equal(p1.data, p2.data)

    st = np.random.rand(4).astype(np.float32)
    act = a1.act(st)
    assert 0 <= act < 3
    
    # 修正：傳入模型狀態字典的列表，而不是代理物件
    client_states = [a1.model.state_dict(), a2.model.state_dict()]
    global_state = fedavg(client_states)
    
    # 將聚合後的模型載入回代理
    a1.model.load_state_dict(global_state)
    a2.model.load_state_dict(global_state)
    
    # 驗證兩個代理的模型參數現在是否完全相同
    for p1, p2 in zip(a1.model.parameters(), a2.model.parameters()):
        assert torch.equal(p1.data, p2.data)
