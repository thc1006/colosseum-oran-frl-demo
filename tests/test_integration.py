import pytest
import torch
import pandas as pd
from pathlib import Path

from colosseum_oran_frl_demo.agents.rl_agent import RLAgent
from colosseum_oran_frl_demo.envs.slice_sim_env import SliceSimEnv
from colosseum_oran_frl_demo.agents.fed_server import fedavg
from colosseum_oran_frl_demo.config import HP, Paths

@pytest.fixture
def dummy_dataframe():
    data = {
        "timestamp": [0, 0, 1, 1, 2, 2, 3, 3],
        "BS_ID": [1, 1, 1, 1, 2, 2, 2, 2],
        "Slice_ID": [0, 1, 0, 1, 0, 1, 0, 1],
        "Throughput_DL_Mbps": [10, 5, 12, 6, 8, 4, 9, 5],
        "Latency_proxy_ms": [15, 8, 14, 7, 20, 10, 18, 9],
    }
    return pd.DataFrame(data)

@pytest.fixture
def client_envs(dummy_dataframe):
    # Create two environments for two clients (gNBs)
    env1 = SliceSimEnv(dummy_dataframe, gnb_id=1)
    env2 = SliceSimEnv(dummy_dataframe, gnb_id=2)
    return [env1, env2]

@pytest.fixture
def global_model():
    # A global model instance
    return RLAgent(state_size=4, action_size=3, device="cpu")

# --- Integration Tests ---

def test_e2e_federated_learning_loop(global_model, client_envs):
    """
    Test a single round of the federated learning loop.
    """
    # 1. Get initial global model weights
    initial_global_weights = {k: v.clone() for k, v in global_model.model.state_dict().items()}

    # 2. Simulate client training
    client_models_states = []
    for env in client_envs:
        client_agent = RLAgent(env.state_size, env.action_size, device="cpu")
        client_agent.model.load_state_dict(initial_global_weights) # Sync with global model

        # Simple training loop
        state, _ = env.reset()
        for _ in range(HP.LOCAL_STEPS): # Use a small number of steps for testing
            action = client_agent.act(state)
            next_state, reward, _, _, _ = env.step(action)
            client_agent.remember(state, action, reward, next_state, False)
            state = next_state
        
        client_agent.replay(batch_size=4)
        client_models_states.append(client_agent.model.state_dict())

    # 3. Perform federated averaging
    new_global_weights = fedavg(client_models_states)
    global_model.model.load_state_dict(new_global_weights)

    # 4. Assertions
    # Check that the model weights have been updated
    final_global_weights = global_model.model.state_dict()
    weights_changed = False
    for key in initial_global_weights:
        if not torch.equal(initial_global_weights[key], final_global_weights[key]):
            weights_changed = True
            break
    assert weights_changed, "Global model weights were not updated after a training round."

def test_multi_client_training_consistency(global_model, client_envs):
    """
    Test that multiple clients can be trained and aggregated.
    """
    # This test is similar to the e2e test but focuses on ensuring
    # the aggregation works correctly with more than one client.
    assert len(client_envs) > 1

    initial_weights = global_model.model.state_dict()
    
    client_states = []
    for env in client_envs:
        agent = RLAgent(env.state_size, env.action_size, device="cpu")
        agent.model.load_state_dict(initial_weights)
        
        state, _ = env.reset()
        agent.remember(state, 0, 1.0, state, False)
        agent.replay(batch_size=1) # Train a little
        client_states.append(agent.model.state_dict())

    aggregated_weights = fedavg(client_states)
    
    # Check that aggregation produces a different model than any of the clients
    for client_state in client_states:
        is_different = any(not torch.equal(aggregated_weights[k], client_state[k]) for k in aggregated_weights)
        assert is_different, "Aggregated model is identical to one of the client models."

def test_config_loading_and_validation(tmp_path):
    """
    Test that the configuration can be loaded and is valid.
    This is a basic test to ensure the config files are structured correctly.
    """
    # We can create a dummy config file or test the existing one.
    # For now, we'll just check the loaded HPs from the actual config.
    assert HP.LR > 0
    assert HP.GAMMA >= 0 and HP.GAMMA <= 1
    assert HP.EPS_DECAY > 0 and HP.EPS_DECAY <= 1
    assert HP.LOCAL_STEPS > 0
    assert Paths.ROOT.exists()
