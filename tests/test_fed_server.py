import torch
import pytest
from src.colosseum_oran_frl_demo.agents.fed_server import fedavg

def test_fedavg_basic_aggregation():
    # Test basic aggregation with two client models
    model_state1 = {"layer1.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "layer1.bias": torch.tensor([1.0, 2.0])}
    model_state2 = {"layer1.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]), "layer1.bias": torch.tensor([3.0, 4.0])}
    client_model_states = [model_state1, model_state2]
    
    averaged_state = fedavg(client_model_states)
    
    expected_weight = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    expected_bias = torch.tensor([2.0, 3.0])
    
    assert torch.allclose(averaged_state["layer1.weight"], expected_weight)
    assert torch.allclose(averaged_state["layer1.bias"], expected_bias)

def test_fedavg_single_client():
    # Test aggregation with a single client model
    model_state = {"layer1.weight": torch.tensor([[1.0, 2.0]]), "layer1.bias": torch.tensor([3.0])}
    client_model_states = [model_state]
    
    averaged_state = fedavg(client_model_states)
    
    assert torch.allclose(averaged_state["layer1.weight"], model_state["layer1.weight"])
    assert torch.allclose(averaged_state["layer1.bias"], model_state["layer1.bias"])

def test_fedavg_empty_input():
    # Test aggregation with an empty list of client models
    client_model_states = []
    averaged_state = fedavg(client_model_states)
    assert averaged_state == {}

def test_fedavg_data_type_preservation():
    # Test that the data type is preserved (or correctly cast to float)
    model_state1 = {"layer1.weight": torch.tensor([[1, 2]], dtype=torch.int32)}
    model_state2 = {"layer1.weight": torch.tensor([[3, 4]], dtype=torch.int32)}
    client_model_states = [model_state1, model_state2]
    
    averaged_state = fedavg(client_model_states)
    
    # The fedavg function explicitly casts to float(), so the output should be float32
    assert averaged_state["layer1.weight"].dtype == torch.float32
    assert torch.allclose(averaged_state["layer1.weight"], torch.tensor([[2.0, 3.0]], dtype=torch.float32))

def test_fedavg_multiple_layers():
    # Test with a more complex model with multiple layers
    model_state1 = {
        "fc1.weight": torch.tensor([[0.1, 0.2]]), "fc1.bias": torch.tensor([0.3]),
        "fc2.weight": torch.tensor([[0.4, 0.5]]), "fc2.bias": torch.tensor([0.6])
    }
    model_state2 = {
        "fc1.weight": torch.tensor([[1.1, 1.2]]), "fc1.bias": torch.tensor([1.3]),
        "fc2.weight": torch.tensor([[1.4, 1.5]]), "fc2.bias": torch.tensor([1.6])
    }
    client_model_states = [model_state1, model_state2]

    averaged_state = fedavg(client_model_states)

    expected_fc1_weight = torch.tensor([[0.6, 0.7]])
    expected_fc1_bias = torch.tensor([0.8])
    expected_fc2_weight = torch.tensor([[0.9, 1.0]])
    expected_fc2_bias = torch.tensor([1.1])

    assert torch.allclose(averaged_state["fc1.weight"], expected_fc1_weight)
    assert torch.allclose(averaged_state["fc1.bias"], expected_fc1_bias)
    assert torch.allclose(averaged_state["fc2.weight"], expected_fc2_weight)
    assert torch.allclose(averaged_state["fc2.bias"], expected_fc2_bias)

def test_fedavg_mismatched_keys():
    # Test aggregation with models having different keys (architectures)
    model_state1 = {"layer1.weight": torch.tensor([[1.0, 2.0]])}
    model_state2 = {"layer2.weight": torch.tensor([[5.0, 6.0]])} # Different key
    client_model_states = [model_state1, model_state2]
    
    with pytest.raises(ValueError, match="Client models have different architectures."):
        fedavg(client_model_states)

def test_fedavg_mismatched_shapes():
    # Test aggregation with models having tensors of different shapes
    model_state1 = {"layer1.weight": torch.tensor([[1.0, 2.0]])}
    model_state2 = {"layer1.weight": torch.tensor([[5.0, 6.0, 7.0]])} # Different shape
    client_model_states = [model_state1, model_state2]
    
    with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
        fedavg(client_model_states)
