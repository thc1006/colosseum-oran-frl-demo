# Colosseum-ORAN-FRL-Demo: Offline Federated RL for O-RAN Slice Management

This repository provides an end-to-end framework for offline federated reinforcement learning (FRL) for dynamic 5G/6G slice resource allocation in O-RAN, using the Colosseum dataset.

## Key Features

- **End-to-End FRL Pipeline:** From data preparation to federated training and evaluation.
- **Realistic O-RAN Simulation:** Utilizes the Colosseum dataset to simulate a real-world O-RAN environment.
- **Modular and Extensible:** Easily customize the environment, agent, and federated learning a`lgorithm.
- **Reproducible Research:** Includes scripts and notebooks for easy reproduction of results.

## Project Structure

```
colosseum-oran-frl-demo/
├── scripts/                # Scripts for data preparation and training
│   ├── make_dataset.py
│   └── train.py
├── src/                    # Source code for the FRL framework
│   └── colosseum_oran_frl_demo/
│       ├── agents/             # RL agents and federated server
│       ├── data/               # Data loading and processing
│       ├── envs/               # O-RAN simulation environment
│       └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── tests/                  # Unit and integration tests
├── parquet file for test/  # Sample data for testing
└── README.md
```

## Getting Started

### 1. Installation

```bash
git clone https://github.com/thc1006/colosseum-oran-frl-demo.git
cd colosseum-oran-frl-demo
pip install -r requirements.txt
```

### 2. Data Preparation

The project includes a sample Parquet file for testing. To use your own data, you will need to convert it to Parquet format. The `scripts/make_dataset.py` script can be used for this purpose.

### 3. Training

To start the federated learning training process, run the following command:

```bash
python scripts/train.py --parquet "./parquet file for test/kpi_traces_final_v_robust.parquet"
```

This will train a federated reinforcement learning model using the sample data. The training artifacts will be saved in the `outputs/` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.