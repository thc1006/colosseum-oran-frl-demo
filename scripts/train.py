# 檔案: scripts/train.py

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import trange
import random
from typing import List, Dict, Any, Tuple

from colosseum_oran_frl_demo.config import HP, Paths
from colosseum_oran_frl_demo.envs.slice_sim_env import SliceSimEnv
from colosseum_oran_frl_demo.agents.rl_agent import RLAgent
from colosseum_oran_frl_demo.agents.fed_server import fedavg
from colosseum_oran_frl_demo.utils.plots import plot_training_results


# 修正：將所有邏輯封裝到 main() 函式中
def main() -> None:
    """
    Main function for offline FRL training.

    This function handles argument parsing, data loading, environment and agent
    initialization, the federated training loop, and saving results.
    """
    # ---------- CLI ---------------------------------------------------------- #
    ap = argparse.ArgumentParser(description="Offline FRL training entrypoint")
    ap.add_argument(
        "--parquet",
        default=str(Paths.PROCESSED / "kpi_traces_final_v_robust.parquet"),
        help="Path to the final processed Parquet file",
    )
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--clients", default="1,2,3")
    ap.add_argument("--num_selected_clients", type=int, default=None, help="Number of clients to select for each round. If None, all clients are used.")
    ap.add_argument("--out", default=str(Paths.OUTPUTS))
    args: argparse.Namespace = ap.parse_args()

    out_dir: Path = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame
    try:
        df = pd.read_parquet(args.parquet)
        if df.empty:
            print(f"Error: Parquet file at {args.parquet} is empty.")
            print("Please ensure the data preparation script generates valid data.")
            return
    except FileNotFoundError:
        print(f"Error: Parquet file not found at {args.parquet}")
        print("Please run the data preparation script first.")
        return
    except Exception as e:
        print(f"An error occurred while reading the parquet file: {e}")
        return

    cid_list: List[int] = list(map(int, args.clients.split(",")))

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    envs: List[SliceSimEnv] = []
    agents: List[RLAgent] = []

    for i in cid_list:
        try:
            env: SliceSimEnv = SliceSimEnv(df, gnb_id=i)
            envs.append(env)
            agents.append(RLAgent(env.state_size, env.action_size, lr=HP.LR, device=DEVICE))
        except ValueError as e:
            print(f"Skipping client {i} due to error: {e}")
            continue

    if not envs:
        print("No environments were successfully initialized. Exiting.")
        return

    history: List[Dict[str, float]] = []
    for rd in trange(args.rounds, desc="Fed Round"):
        # Client selection
        selected_agents: List[RLAgent]
        selected_envs: List[SliceSimEnv]
        if args.num_selected_clients and args.num_selected_clients < len(agents):
            selected_clients_indices: List[int] = random.sample(range(len(agents)), args.num_selected_clients)
            selected_agents = [agents[i] for i in selected_clients_indices]
            selected_envs = [envs[i] for i in selected_clients_indices]
        else:
            selected_agents = agents
            selected_envs = envs

        client_model_states: List[Dict[str, torch.Tensor]] = [
            ag.model.state_dict() for ag in selected_agents
        ]

        global_model_state: Dict[str, torch.Tensor] = fedavg(client_model_states)

        for ag in selected_agents:
            ag.model.load_state_dict(global_model_state)

        round_rewards: List[float] = []
        round_losses: List[float] = []

        for ag, env in zip(selected_agents, selected_envs):
            state: np.ndarray
            ep_reward: float
            state, _ = env.reset()
            ep_reward = 0.0

            for step in range(HP.LOCAL_STEPS):
                act: int = ag.act(state)
                nst: np.ndarray
                rwd: float
                term: bool
                trunc: bool
                nst, rwd, term, trunc, _ = env.step(act)
                ag.remember(state, act, rwd, nst, term or trunc)

                loss: Any = ag.replay(64)
                if loss is not None:
                    round_losses.append(loss)

                ep_reward += rwd
                state = nst
                if term or trunc:
                    break

            if HP.LOCAL_STEPS > 0:
                # Use actual steps taken for average reward calculation
                round_rewards.append(ep_reward / (step + 1) if (step + 1) > 0 else 0.0)

        avg_reward: float = float(np.mean(round_rewards)) if round_rewards else 0.0
        avg_loss: float = float(np.mean(round_losses)) if round_losses else 0.0

        history.append(
            {
                "round": float(rd),
                "reward": avg_reward,
                "loss": avg_loss,
                "epsilon": float(np.mean([ag.epsilon for ag in selected_agents]) if selected_agents else 0.0),
            }
        )
        print(
            f"Round {rd+1}/{args.rounds} - Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}"
        )

    # Save history and model
    history_df: pd.DataFrame = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)
    if agents:
        torch.save(agents[0].model.state_dict(), out_dir / "global_model.pt")
    else:
        print("No agents were initialized, skipping model save.")

    # Plot results
    if not history_df.empty:
        plot_training_results(history_df["reward"], history_df["loss"])

    with open(out_dir / "config.json", "w") as fp:
        config_data: Dict[str, Any] = vars(args)
        # Convert Path objects to strings for JSON serialization
        config_data["HP"] = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in HP.__dict__.items()
            if not k.startswith("__")
        }
        config_data["out"] = str(config_data["out"])
        config_data["parquet"] = str(config_data["parquet"])
        json.dump(config_data, fp, indent=2)

    print(f"Finished {args.rounds} rounds – artifacts saved to {out_dir.resolve()}")


# 修正：增加 if __name__ == "__main__": 區塊
if __name__ == "__main__":
    main()
