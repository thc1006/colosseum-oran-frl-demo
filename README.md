<h1 align="center">
  Colosseum-ORAN-FRL-Demo<br>
  Offline Federated RL for O-RAN Slice Management
</h1>

<p align="center">
  <b>Colosseum-ORAN-FRL-Demo</b> is an end-to-end research starter-kit that turns<br>
  <a href="https://github.com/wineslab/colosseum-oran-coloran-dataset">Colosseum ORAN KPI traces</a>
  into a reproducible, <em>offline</em> Federated Reinforcement Learning (FRL) pipeline  
  for dynamic 5G/6G slice resource allocation.
</p>

<p align="center">
  <a href="https://github.com/thc1006/colosseum-oran-frl-demo/blob/main/LICENSE"><img
    src="https://img.shields.io/badge/License-MIT-green"
    alt="license"/></a>
  <a href="https://github.com/thc1006/colosseum-oran-frl-demo/actions/workflows/ci.yml">
    <img src="https://github.com/thc1006/colosseum-oran-frl-demo/actions/workflows/ci.yml/badge.svg"
      alt="CI Status"/></a>
  <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /></a>
  <img src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11|%203.12-blue"/>
  <img src="https://img.shields.io/badge/torch-2.x-red"/>
  <img src="https://img.shields.io/badge/PyTorch-1.10.0%2B-orange"/>
</p>

---

## ✨ Key Features
| Category | What you get | Why it matters |
|-----------|--------------|----------------|
| **Data-pipeline** | `make_dataset.py` converts raw Colosseum CSV → partitioned Parquet with metadata columns (`sched`, `tr`, `exp`, `bs`) | 20× faster load & reusable by any ML task |
| **Offline simulator** | `SliceSimEnv` replays KPI traces as a Gym-like env with 3-action discrete resource split (70/30, 50/50, 30/70) | Rapid prototyping without emulation bed |
| **Federated RL** | Minimal DQN + FedAvg loop (`train.py`) where each gNB = client | Mirrors multi-site xApp deployment in O-RAN |
| **Reproducibility** | One-command Colab/Binder notebooks + GitHub Actions nbmake | Reviewers & teammates re-run in minutes |
| **Modular package** | `src/colosseum_oran_frl_demo` ready for `pip install -e .` | Swap env/agent without touching notebooks |

> The README structure re-uses best practices distilled from HuggingFace Transformers, PyTorch, and the “Best-README-Template”.

---

## 🗺️  Project Layout
```
colosseum-oran-frl-demo/
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── .github/workflows/ci.yml
│
├── scripts/
│   ├── make_dataset.py
│   └── train.py
│
├── src/
│   └── colosseum_oran_frl_demo/
│       ├── init.py
│       ├── config.py
│       ├── data/
│       ├── envs/
│       ├── agents/
│       └── utils/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_frl_training.ipynb
│
└── tests/
├── test_dataset.py
├── test_env.py
└── test_agents.py

````

### Murmur: Why I Chose Google Colab
In this project, I manually downloaded the [**Original Dataset**](https://github.com/wineslab/colosseum-oran-coloran-dataset) (ZIP archive) to my local machine and then uploaded it to my personal Google Drive, because Google Colab imposes daily bandwidth limits on `git clone` operations that make a one-time download of such a large dataset impractical. By mounting my Drive within Colab, I ensured both data integrity and ease of access, and have confirmed that the workflow runs successfully on both my local environment and in the Colab notebook. Moreover, since my laptop’s only got GTX 1050 Laptop version GPU could not deliver sufficient performance (resulting in unacceptably slow processing). That’s why I used Colab’s free T4 GPU, which provides better computational power for the demands of this project.

### If you want to execute Notebooks on Google Colab like I do:
1. Download the colosseum-oran-coloran-dataset(ZIP archive). Unzip it and upload to ur Google Drive.
2. Run [**01_Data_Preparation.ipynb**](https://colab.research.google.com/drive/1OIAcJt7oQWsaMwzed1p0HV2Olrcg5Y14) for Clean and Format Dataset (Remember change Dataset path to ur own !!!).
3. Run [**02_FRL_Training.ipynb**](https://colab.research.google.com/drive/1z8N3Ex1l2outgCnk6yXumGuJERHP8Lqv) for Federated Reinforcement Learning Sitmulation.

> Remember to **Save a Copy to ur own Drive** before you start executing Cells!



## 🚀 Quick Start

> Warning!!! This section is workin in progress... Might not stable.
> There is a [**Google Colab version of the previous block**](https://github.com/thc1006/colosseum-oran-frl-demo/edit/main/README.md#if-you-want-to-execute-notebooks-on-google-colab-like-i-do), so you can work through that first.

### 0. Install
```bash
git clone [https://github.com/thc1006/colosseum-oran-frl-demo.git](https://github.com/thc1006/colosseum-oran-frl-demo.git)
cd colosseum-oran-frl-demo
python3 -m venv .venv && source .venv/bin/activate # Win: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 1. Data Prep (Requires external dataset)

First, download or locate the `colosseum-oran-coloran-dataset` raw CSV files. Then, run the script pointing to the raw data directory and your desired output location.

```bash
python scripts/make_dataset.py \
  --raw   /path/to/your/colosseum-oran-coloran-dataset/rome_static_medium \
  --out   src/colosseum_oran_frl_demo/data/processed
```

### 2. Offline FRL Training

This script will automatically use the processed data from the default path specified in `src/colosseum_oran_frl_demo/config.py`.

```bash
python scripts/train.py --rounds 10 --clients 1,2,6 --out outputs
```

Training artifacts will be saved in the `outputs/` directory.

### 3. Notebook Walkthrough

For a step-by-step guide, open the notebooks:
1. `notebooks/01_data_preparation.ipynb`
2. `notebooks/02_frl_training.ipynb`

---

## 🏗️  Roadmap

* [ ] Continuous-action SAC / PPO
* [ ] FedProx & FedNova baselines
* [ ] Kubernetes Helm chart for Near-RT RIC deployment
* [ ] Full Colosseum over-the-air live demo


## 🤝 Contributing

1. Fork → Create branch → `make lint test` → PR.
2. New features **must** include unit tests and docstrings (Google style).
3. Large models / data  ➜ push via Git LFS.

Guidelines are adapted from the
[Awsome-Readme collection](https://github.com/matiassingers/awesome-readme).

## 📜 License

Distributed under the MIT License. See **[LICENSE](https://github.com/thc1006/colosseum-oran-frl-demo/blob/main/LICENSE)** for details.


## 🙏 Acknowledgements

This project was made possible by the foundational work of the [**WiNES Lab at Northeastern University**](https://ece.northeastern.edu/wineslab/), who created and open-sourced the [**Colosseum O-RAN Dataset**](https://openrangym.com/datasets/colosseum-coloran-dataset).

This large-scale, high-fidelity dataset of O-RAN KPI traces provided an invaluable foundation for this research, enabling meaningful offline reinforcement learning experiments that closely mirror real-world network scenarios. Without access to such high-quality, open data, this work would not have been feasible.

We extend our sincerest gratitude to the WiNES Lab for their significant contributions to the open-access wireless research community.

## 📜 Citation

If you use this repository in your academic work, please cite it as follows:

```bibtex
@misc{colosseum_oran_frl_demo,
  author       = {Tsai, Hsiu-Chi and Contributors},
  title        = {{Colosseum-ORAN-FRL-Demo}: Offline Federated RL for O-RAN Slicing},
  year         = {2025},
  url          = {[https://github.com/thc1006/colosseum-oran-frl-demo](https://github.com/thc1006/colosseum-oran-frl-demo)},
}
```

Furthermore, to properly credit the [original dataset](https://github.com/wineslab/colosseum-oran-coloran-dataset), we strongly recommend citing the official Colosseum paper in your work as well:

```bibtex
@article{polese2022coloran,
  author  = {M. Polese and L. Bonati and S. D'Oro and S. Basagni and T. Melodia},
  title   = {ColO-RAN: Developing Machine Learning-based xApps for Open RAN Closed-loop Control},
  journal = {IEEE Trans. Mobile Comput.},
  year    = {2022}
}
```


## 🙋‍♀️  Contact

* **Issues** · GitHub Issues tab
* **Email**  · [hctsai@linux.com](mailto:hctsai@linux.com)
* **Twitter/X** · [@thc1006](https://x.com/@thc1006)

Happy slicing & federating! ! 🍰 ;)
