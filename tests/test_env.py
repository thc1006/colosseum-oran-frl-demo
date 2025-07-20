import numpy as np
import pandas as pd
import pytest
from colosseum_oran_frl_demo.envs.slice_sim_env import SliceSimEnv

@pytest.fixture
def sample_dataframe():
    # A more comprehensive sample DataFrame for testing
    data = {
        "timestamp": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "BS_ID": [1, 1, 2, 1, 1, 2, 1, 1, 2],
        "Slice_ID": [0, 2, 0, 0, 2, 0, 0, 2, 0],
        "Throughput_DL_Mbps": [10, 5, 8, 12, 6, 9, 15, 7, 10],
        "Latency_proxy_ms": [15, 8, 12, 14, 7, 11, 13, 6, 9],
    }
    return pd.DataFrame(data)

@pytest.fixture
def zero_kpi_dataframe():
    data = {
        "timestamp": [0, 0, 1, 1],
        "BS_ID": [1, 1, 1, 1],
        "Slice_ID": [0, 2, 0, 2],
        "Throughput_DL_Mbps": [0, 0, 0, 0],
        "Latency_proxy_ms": [0, 0, 0, 0],
    }
    return pd.DataFrame(data)

@pytest.fixture
def env(sample_dataframe):
    return SliceSimEnv(sample_dataframe, gnb_id=1)

# --- Initialization Tests ---
def test_env_initialization_valid(sample_dataframe):
    env = SliceSimEnv(sample_dataframe, gnb_id=1, slice_ids=(0, 2))
    assert env.df_data["BS_ID"].nunique() == 1
    assert env.df_data["BS_ID"].iloc[0] == 1
    assert env.slices == (0, 2)
    assert len(env.ts) > 0
    assert env.cursor == 0

def test_env_initialization_invalid_gnb_id(sample_dataframe):
    with pytest.raises(ValueError, match="No data found for gNB ID 99 in the provided DataFrame."):
        SliceSimEnv(sample_dataframe, gnb_id=99)

def test_env_initialization_empty_dataframe():
    empty_df = pd.DataFrame(columns=["timestamp", "BS_ID", "Slice_ID", "Throughput_DL_Mbps", "Latency_proxy_ms"])
    with pytest.raises(ValueError, match="No data found for gNB ID 1 in the provided DataFrame."):
        SliceSimEnv(empty_df, gnb_id=1)

# --- _prep method tests ---
def test_env_prep_sorts_data(sample_dataframe):
    # Create a shuffled dataframe to ensure sorting works
    shuffled_df = sample_dataframe.sample(frac=1).reset_index(drop=True)
    env = SliceSimEnv(shuffled_df, gnb_id=1)
    # Check if timestamps are sorted
    assert list(env.df_data["timestamp"]) == sorted(list(env.df_data["timestamp"]))
    assert len(env.ts) == shuffled_df[shuffled_df["BS_ID"] == 1]["timestamp"].nunique()

# --- _state_from_row method tests ---
def test_env_state_from_row_valid_data(env, sample_dataframe):
    sub = sample_dataframe[(sample_dataframe["BS_ID"] == 1) & (sample_dataframe["timestamp"] == 0)]
    state = env._state_from_row(sub)
    expected_state = np.array([10.0, 15.0, 5.0, 8.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(state, expected_state)

def test_env_state_from_row_missing_slice(env, sample_dataframe):
    # Create a sub-dataframe missing data for slice 2
    sub = sample_dataframe[(sample_dataframe["BS_ID"] == 1) & (sample_dataframe["timestamp"] == 0) & (sample_dataframe["Slice_ID"] == 0)]
    state = env._state_from_row(sub)
    # Expect 0.0 for missing slice's tput and lat
    expected_state = np.array([10.0, 15.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(state, expected_state)

# --- _compute_reward method tests ---
def test_env_compute_reward(env):
    state = np.array([10.0, 15.0, 5.0, 8.0], dtype=np.float32) # tput0=10, lat0=15, tput2=5, lat2=8
    reward = env._compute_reward(state)
    expected_reward = (0.3 * 10.0) - (0.7 * 8.0)  # 3.0 - 5.6 = -2.6
    assert reward == pytest.approx(expected_reward)

def test_env_compute_reward_zero_values(env):
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    reward = env._compute_reward(state)
    assert reward == 0.0

@pytest.mark.parametrize(
    "tput0, lat0, tput2, lat2, expected_reward",
    [
        (10.0, 15.0, 5.0, 8.0, (0.3 * 10.0) - (0.7 * 8.0)),  # Original test case
        (0.0, 0.0, 0.0, 0.0, 0.0),  # All zeros
        (100.0, 1.0, 100.0, 1.0, (0.3 * 100.0) - (0.7 * 1.0)),  # High throughput, low latency
        (1.0, 100.0, 1.0, 100.0, (0.3 * 1.0) - (0.7 * 100.0)),  # Low throughput, high latency
        (50.0, 5.0, 0.0, 0.0, (0.3 * 50.0) - (0.7 * 0.0)),  # One slice active, other inactive
        (0.0, 0.0, 50.0, 5.0, (0.3 * 0.0) - (0.7 * 5.0)),  # Other slice active, one inactive
    ],
)
def test_env_compute_reward_parametrized(env, tput0, lat0, tput2, lat2, expected_reward):
    state = np.array([tput0, lat0, tput2, lat2], dtype=np.float32)
    reward = env._compute_reward(state)
    assert reward == pytest.approx(expected_reward)

# --- Gym-like API Tests ---
def test_env_reset(env, sample_dataframe):
    state, info = env.reset()
    expected_state = np.array([10.0, 15.0, 5.0, 8.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(state, expected_state)
    assert info == {}
    assert env.cursor == 0

def test_env_step_transition(env, sample_dataframe):
    # Initial state at timestamp 0
    env.reset()
    # Step to timestamp 1
    state, reward, terminated, truncated, info = env.step(0) # Action doesn't matter for state transition

    expected_state_ts1 = np.array([12.0, 14.0, 6.0, 7.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(state, expected_state_ts1)
    assert reward == pytest.approx((0.3 * 12.0) - (0.7 * 7.0)) # 3.6 - 4.9 = -1.3
    assert not terminated
    assert not truncated
    assert info["timestamp"] == 1
    assert env.cursor == 1

def test_env_step_wraps_around(env, sample_dataframe):
    # Fast forward cursor to the last timestamp
    env.cursor = len(env.ts) - 1
    last_timestamp = env.ts[env.cursor]

    state, reward, terminated, truncated, info = env.step(0)

    # After stepping from the last timestamp, it should wrap around to the first timestamp (0)
    first_timestamp_data = sample_dataframe[(sample_dataframe["BS_ID"] == 1) & (sample_dataframe["timestamp"] == 0)]
    expected_state_wrapped = env._state_from_row(first_timestamp_data)

    np.testing.assert_array_almost_equal(state, expected_state_wrapped)
    assert info["timestamp"] == 0
    assert env.cursor == 0

def test_env_with_zero_kpis(zero_kpi_dataframe):
    env = SliceSimEnv(zero_kpi_dataframe, gnb_id=1)
    state, _ = env.reset()
    expected_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(state, expected_state)

    state, reward, _, _, _ = env.step(0)
    np.testing.assert_array_almost_equal(state, expected_state)
    assert reward == 0.0

# --- Properties Tests ---
def test_env_action_size(env):
    assert env.action_size == 3

def test_env_state_size(env):
    assert env.state_size == 4
