# 檔案: tests/test_agents.py

import numpy as np
import torch
from colosseum_oran_frl_demo.agents.rl_agent import RLAgent
from colosseum_oran_frl_demo.agents.fed_server import fedavg
from typing import List, Dict

def test_fedavg_empty_input() -> None:
    """Test fedavg with an empty list of client states."""
    assert fedavg([]) == {}

def test_agent_and_fedavg() -> None:
    """Test RLAgent and fedavg functionality."""
    state_size = 4
    action_size = 3
    a1 = RLAgent(state_size, action_size)
    a2 = RLAgent(state_size, action_size)

    # Ensure initial weights are different
    for p1, p2 in zip(a1.model.parameters(), a2.model.parameters()):
        p2.data = torch.randn_like(p1.data)
        assert not torch.equal(p1.data, p2.data)

    # Test RLAgent.act()
    st = np.random.rand(state_size).astype(np.float32)
    act = a1.act(st)
    assert 0 <= act < action_size

    # Test RLAgent.remember() and RLAgent.replay()
    initial_memory_len = len(a1.memory)
    for _ in range(64):
        a1.remember(st, act, 0.1, st, False)
    assert len(a1.memory) == initial_memory_len + 64
    
    # Replay should not raise an error and loss should be calculated
    a1.replay(64)

    # Test fedavg
    client_states: List[Dict[str, torch.Tensor]] = [a1.model.state_dict(), a2.model.state_dict()]
    global_state: Dict[str, torch.Tensor] = fedavg(client_states)

    # Load aggregated model back to agents
    a1.model.load_state_dict(global_state)
    a2.model.load_state_dict(global_state)

    # Verify that model parameters are now identical
    for p1, p2 in zip(a1.model.parameters(), a2.model.parameters()):
        assert torch.equal(p1.data, p2.data)
