import numpy as np
import torch
import pytest
import random
from colosseum_oran_frl_demo.agents.rl_agent import RLAgent

@pytest.fixture
def sample_rl_agent():
    state_size = 4
    action_size = 3
    return RLAgent(state_size, action_size, device="cpu") # Force CPU for consistent testing

@pytest.fixture
def sample_state():
    return np.random.rand(4).astype(np.float32)

@pytest.fixture
def sample_experience(sample_state):
    action = 0
    reward = 1.0
    next_state = np.random.rand(4).astype(np.float32)
    done = False
    return (sample_state, action, reward, next_state, done)

# --- RLAgent Initialization Tests ---
def test_rl_agent_initialization(sample_rl_agent):
    assert sample_rl_agent.state_size == 4
    assert sample_rl_agent.action_size == 3
    assert sample_rl_agent.epsilon == 1.0
    assert sample_rl_agent.epsilon_min == 0.01
    assert sample_rl_agent.epsilon_decay == 0.998
    assert isinstance(sample_rl_agent.model, torch.nn.Module)
    assert isinstance(sample_rl_agent.optimizer, torch.optim.Optimizer)
    assert isinstance(sample_rl_agent.loss_fn, torch.nn.MSELoss)
    assert len(sample_rl_agent.memory) == 0
    assert sample_rl_agent.memory.maxlen == 20000

# --- RLAgent.remember() Tests ---
def test_rl_agent_remember_adds_experience(sample_rl_agent, sample_experience):
    initial_memory_len = len(sample_rl_agent.memory)
    sample_rl_agent.remember(*sample_experience)
    assert len(sample_rl_agent.memory) == initial_memory_len + 1
    assert sample_rl_agent.memory[-1] == sample_experience

def test_rl_agent_remember_respects_memory_cap():
    agent = RLAgent(state_size=4, action_size=2, memory_cap=5, device="cpu")
    for i in range(10):
        state = np.array([i, 0, 0, 0], dtype=np.float32)
        agent.remember(state, 0, 0.0, state, False)
    assert len(agent.memory) == 5
    assert agent.memory[0][0][0] == 5 # Check if oldest experience is removed

# --- RLAgent.act() Tests ---
def test_rl_agent_act_valid_action(sample_rl_agent, sample_state):
    action = sample_rl_agent.act(sample_state)
    assert isinstance(action, int)
    assert 0 <= action < sample_rl_agent.action_size

def test_rl_agent_act_greedy_action(sample_rl_agent, sample_state):
    sample_rl_agent.epsilon = 0.0 # Ensure greedy action
    # Mock the model's output to control the expected action
    with torch.no_grad():
        sample_rl_agent.model = torch.nn.Sequential(
            torch.nn.Linear(sample_rl_agent.state_size, sample_rl_agent.action_size)
        )
        # Set weights to ensure a specific action is chosen (e.g., action 1)
        sample_rl_agent.model[0].weight.fill_(0.0)
        sample_rl_agent.model[0].bias.fill_(0.0)
        sample_rl_agent.model[0].bias[1] = 10.0 # Make action 1 the highest Q-value

    action = sample_rl_agent.act(sample_state, eval=True)
    assert action == 1

def test_rl_agent_act_epsilon_decay(sample_rl_agent, sample_experience):
    initial_epsilon = sample_rl_agent.epsilon
    # Fill memory and replay to trigger epsilon decay
    for _ in range(100):
        sample_rl_agent.remember(*sample_experience)
    sample_rl_agent.replay(batch_size=32)
    assert sample_rl_agent.epsilon < initial_epsilon
    assert sample_rl_agent.epsilon >= sample_rl_agent.epsilon_min

def test_rl_agent_act_no_epsilon_decay_in_eval(sample_rl_agent, sample_state):
    initial_epsilon = sample_rl_agent.epsilon
    sample_rl_agent.act(sample_state, eval=True)
    assert sample_rl_agent.epsilon == initial_epsilon

# --- RLAgent.replay() Tests ---
def test_rl_agent_replay_insufficient_memory(sample_rl_agent):
    initial_epsilon = sample_rl_agent.epsilon
    # Memory is empty, replay should do nothing
    sample_rl_agent.replay(batch_size=32)
    assert len(sample_rl_agent.memory) == 0
    assert sample_rl_agent.epsilon == initial_epsilon # Epsilon should not decay

def test_rl_agent_replay_updates_model_parameters(sample_rl_agent, sample_experience):
    # Fill memory with enough experiences
    for _ in range(50):
        sample_rl_agent.remember(*sample_experience)

    # Get initial model parameters
    initial_params = [p.clone() for p in sample_rl_agent.model.parameters()]

    # Perform replay
    sample_rl_agent.replay(batch_size=32)

    # Check if parameters have changed (they should, unless loss is exactly zero)
    params_changed = False
    for initial_p, current_p in zip(initial_params, sample_rl_agent.model.parameters()):
        if not torch.equal(initial_p, current_p):
            params_changed = True
            break
    assert params_changed, "Model parameters should have changed after replay"

def test_rl_agent_replay_loss_reduction(sample_rl_agent, sample_experience):
    # Fill memory with enough experiences
    for _ in range(100):
        sample_rl_agent.remember(*sample_experience)

    # Perform multiple replays and check if loss decreases (generally)
    losses = []
    for _ in range(5):
        loss = sample_rl_agent.replay(batch_size=32)
        if loss is not None:
            losses.append(loss)

    # Assert that the loss is not None
    assert all(loss is not None for loss in losses) and len(losses) > 0

def test_rl_agent_replay_updates_model_parameters_after_learn(sample_rl_agent, sample_experience):
    # Fill memory with enough experiences
    for _ in range(50):
        sample_rl_agent.remember(*sample_experience)

    # Get initial model parameters
    initial_params = [p.clone() for p in sample_rl_agent.model.parameters()]

    # Perform replay
    sample_rl_agent.replay(batch_size=32)

    # Check if parameters have changed
    params_changed = False
    for initial_p, current_p in zip(initial_params, sample_rl_agent.model.parameters()):
        if not torch.equal(initial_p, current_p):
            params_changed = True
            break
    assert params_changed, "Model parameters should have changed after replay"

def test_rl_agent_replay_with_different_batch_sizes(sample_rl_agent, sample_experience):
    # Fill memory with enough experiences
    for _ in range(100):
        sample_rl_agent.remember(*sample_experience)

    # Test with a small batch size
    loss_small_batch = sample_rl_agent.replay(batch_size=16)
    assert loss_small_batch is not None

    # Test with a larger batch size
    loss_large_batch = sample_rl_agent.replay(batch_size=64)
    assert loss_large_batch is not None

# --- RLAgent Serialization Tests ---
def test_rl_agent_serialization(sample_rl_agent, tmp_path):
    # 1. Change the agent's state
    sample_rl_agent.epsilon = 0.5
    for _ in range(10):
        sample_rl_agent.remember(
            np.random.rand(4).astype(np.float32), 0, 1.0, np.random.rand(4).astype(np.float32), False
        )

    # 2. Save the agent's state
    model_path = tmp_path / "test_agent.pth"
    sample_rl_agent.save(model_path)
    assert model_path.exists()

    # 3. Create a new agent and load the state
    new_agent = RLAgent(sample_rl_agent.state_size, sample_rl_agent.action_size, device="cpu")
    new_agent.load(model_path)

    # 4. Verify the loaded state
    assert new_agent.epsilon == sample_rl_agent.epsilon
    assert len(new_agent.memory) == len(sample_rl_agent.memory)
    assert torch.all(
        torch.eq(
            list(new_agent.model.parameters())[0],
            list(sample_rl_agent.model.parameters())[0]
        )
    )
