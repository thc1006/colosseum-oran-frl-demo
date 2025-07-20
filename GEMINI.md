# Colosseum O-RAN Federated Reinforcement Learning Project

## Project Context
This is an offline federated reinforcement learning (FRL) framework for O-RAN slice resource management using Colosseum traces. The project implements a multi-client DQN+FedAvg pipeline for 5G/6G network slice optimization.

## Core Architecture
- **Environment**: `SliceSimEnv` - Gym-style KPI trace replay with 3-action discrete resource allocation
- **Agent**: `RLAgent` - Lightweight DQN with epsilon-greedy exploration and experience replay
- **Federation**: `fedavg` function implementing FedAvg algorithm for model aggregation
- **Data Pipeline**: CSV→Parquet conversion with 20x performance improvement

## Critical Issues to Address
1. **Federated Learning Functionality**: Ensure proper client-server communication and model aggregation
2. **Error Handling**: Implement robust error handling throughout the training pipeline
3. **Data Validation**: Add comprehensive data integrity checks
4. **Test Coverage**: Expand unit test coverage for all core components

## Technical Constraints
- Python 3.9-3.12 compatibility
- PyTorch 2.1+ with CUDA support
- Parquet data format for efficient I/O
- Modular architecture supporting easy component swapping

## Success Criteria
- Federated training loop executes without errors
- Multiple clients can train and aggregate models successfully
- Performance metrics show convergence over training rounds
- All tests pass with adequate coverage
