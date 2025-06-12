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
  <img src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue"/>
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
In this project, I manually downloaded the original dataset (ZIP archive) to my local machine and then uploaded it to my personal Google Drive, because Google Colab imposes daily bandwidth limits on `git clone` operations that make a one-time download of such a large dataset impractical. By mounting my Drive within Colab, I ensured both data integrity and ease of access, and have confirmed that the workflow runs successfully on both my local environment and in the Colab notebook. Moreover, since my laptop’s only got GTX 1050 Laptop version GPU could not deliver sufficient performance (resulting in unacceptably slow processing). That’s why I used Colab’s free T4 GPU, which provides better computational power for the demands of this project.



## 🚀 Quick Start

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
# 修正：指令範例與預設路徑對齊，更具指導性
python scripts/make_dataset.py \
  --raw   /path/to/your/colosseum-oran-coloran-dataset/rome_static_medium \
  --out   src/colosseum_oran_frl_demo/data/processed
```

### 2. Offline FRL Training

This script will automatically use the processed data from the default path specified in `src/colosseum_oran_frl_demo/config.py`.

```bash
# 修正：簡化指令，因為腳本已有預設路徑
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
[Awsome-Readme collection]([github.com][1]).

## 📜 License

Distributed under the MIT License. See **[LICENSE](https://github.com/thc1006/colosseum-oran-frl-demo/blob/main/LICENSE)** for details.


## 🙏 Acknowledgements

**Wireless Networks and Embedded Systems Lab (WiNES) Lab** at Northeastern University for releasing the dataset ([github.com][4]).
README structure inspired by community best-practice articles on badges and documentation ([daily.dev][3]) and by high-impact FL projects ([github.com][2]).

## Citation

If you use this repo in academic work, please cite:

```bibtex
@misc{colosseum_oran_frl_demo,
  author       = {Tsai, Hsiu-Chi and Contributors},
  title        = {{Colosseum-ORAN-FRL-Demo}: Offline Federated RL for O-RAN Slicing},
  year         = 2025,
  url          = {https://github.com/yourname/colosseum-oran-frl-demo},
}
```

If you use [this dataset](https://github.com/wineslab/colosseum-oran-coloran-dataset)（wineslab's colosseum-oran-coloran-dataset）in academic work, please cite:

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

[1]: https://github.com/matiassingers/awesome-readme?utm_source=chatgpt.com "matiassingers/awesome-readme - GitHub"
[2]: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/README.md "README.md - AshwinRJ/Federated-Learning-PyTorch - GitHub"
[3]: https://daily.dev/blog/readme-badges-github-best-practices "Readme Badges GitHub: Best Practices - Daily.dev"
[4]: https://github.com/wineslab/colosseum-oran-coloran-dataset "Colosseum O-RAN ColORAN Dataset - GitHub"
