# ─── tests/test_agents.py ───
import numpy as np
from colosseum_oran_frl_demo.agents.rl_agent import RLAgent
from colosseum_oran_frl_demo.agents.fed_server import fedavg


def test_agent_and_fedavg():
    a1 = RLAgent(4, 3)
    a2 = RLAgent(4, 3)
    st = np.random.rand(4).astype(np.float32)
    act = a1.act(st)
    assert 0 <= act < 3
    fedavg([a1, a2])
    for p1, p2 in zip(a1.model.parameters(), a2.model.parameters()):
        assert p1.shape == p2.shape
