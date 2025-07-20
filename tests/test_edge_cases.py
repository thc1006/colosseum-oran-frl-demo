import pytest
import torch
import pandas as pd
import numpy as np

from colosseum_oran_frl_demo.agents.rl_agent import RLAgent
from colosseum_oran_frl_demo.envs.slice_sim_env import SliceSimEnv
from colosseum_oran_frl_demo.agents.fed_server import fedavg

@pytest.fixture
def dummy_dataframe():
    data = {
        "timestamp": [0, 0, 1, 1],
        "BS_ID": [1, 1, 1, 1],
        "Slice_ID": [0, 1, 0, 1],
        "Throughput_DL_Mbps": [10, 5, 12, 6],
        "Latency_proxy_ms": [15, 8, 14, 7],
    }
    return pd.DataFrame(data)

# --- Edge Case Tests ---

def test_training_with_empty_dataframe():
    """
    Test that the system handles an empty DataFrame gracefully.
    """
    empty_df = pd.DataFrame(columns=["timestamp", "BS_ID", "Slice_ID", "Throughput_DL_Mbps", "Latency_proxy_ms"])
    with pytest.raises(ValueError, match="No data found for gNB ID 1"):
        SliceSimEnv(empty_df, gnb_id=1)

def test_client_dropout_scenario(dummy_dataframe):
    """
    Test the federated averaging with one client dropping out (not submitting a model).
    """
    # 1. Setup clients
    env1 = SliceSimEnv(dummy_dataframe, gnb_id=1)
    env2 = SliceSimEnv(dummy_dataframe, gnb_id=1) # Same data for simplicity
    
    agent1 = RLAgent(env1.state_size, env1.action_size, device="cpu")
    agent2 = RLAgent(env2.state_size, env2.action_size, device="cpu")

    # 2. Train both agents
    state1, _ = env1.reset()
    agent1.remember(state1, 0, 1.0, state1, False)
    agent1.replay(batch_size=1)

    state2, _ = env2.reset()
    agent2.remember(state2, 0, 1.0, state2, False)
    agent2.replay(batch_size=1)

    # 3. Simulate dropout: only agent1's model is submitted
    client_models_states = [agent1.model.state_dict()]
    
    # 4. Perform federated averaging
    new_global_weights = fedavg(client_models_states)

    # 5. Assertions
    # The new global model should be identical to the one that participated
    assert all(torch.equal(new_global_weights[k], agent1.model.state_dict()[k]) for k in new_global_weights)
    assert not all(torch.equal(new_global_weights[k], agent2.model.state_dict()[k]) for k in new_global_weights)

def test_model_divergence_detection():
    """
    Test that the aggregation function can handle divergent models (NaN or inf weights).
    """
    model_state1 = {"layer.weight": torch.tensor([[1.0, 2.0]])}
    
    # Create a model with NaN weights
    diverged_weights = torch.tensor([[float('nan'), 4.0]])
    model_state2_nan = {"layer.weight": diverged_weights}

    # Create a model with inf weights
    diverged_weights_inf = torch.tensor([[float('inf'), 4.0]])
    model_state2_inf = {"layer.weight": diverged_weights_inf}

    # Test with NaN
    with pytest.raises(ValueError, match="Detected NaN in model weights from client"):
        fedavg([model_state1, model_state2_nan])

    # Test with Infinity
    with pytest.raises(ValueError, match="Detected Inf in model weights from client"):
        fedavg([model_state1, model_state2_inf])

def test_resource_exhaustion_memory():
    """
    Test the agent's memory limit to prevent resource exhaustion.
    """
    agent = RLAgent(state_size=4, action_size=2, memory_cap=10, device="cpu")
    
    for i in range(20):
        state = np.random.rand(4).astype(np.float32)
        agent.remember(state, 0, 1.0, state, False)
    
    assert len(agent.memory) == 10, "Memory size should not exceed the specified capacity."
