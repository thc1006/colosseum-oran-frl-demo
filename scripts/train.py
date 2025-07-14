# 檔案: scripts/train.py

from __future__ import annotations
import argparse, json, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import trange

from colosseum_oran_frl_demo.config import HP, Paths
from colosseum_oran_frl_demo.envs.slice_sim_env import SliceSimEnv
from colosseum_oran_frl_demo.agents.rl_agent import RLAgent
from colosseum_oran_frl_demo.agents.fed_server import fedavg
from colosseum_oran_frl_demo.utils.plots import plot_training_results


# 修正：將所有邏輯封裝到 main() 函式中
def main():
    # ---------- CLI ---------------------------------------------------------- #
    ap = argparse.ArgumentParser(description="""Offline FRL training entrypoint""")
    ap.add_argument(
        """--parquet""",
        default=str(Paths.PROCESSED / """kpi_traces_final_v_robust.parquet"""),
        help="""Path to the final processed Parquet file""",
    )
    ap.add_argument("""--rounds""", type=int, default=10)
    ap.add_argument("""--clients""", default="""1,2,3""")
    ap.add_argument("""--out""", default=str(Paths.OUTPUTS))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_parquet(args.parquet)
    except FileNotFoundError:
        print(f"""Error: Parquet file not found at {args.parquet}""")
        print("""Please run the data preparation script first.""")
        return

    cid_list = list(map(int, args.clients.split(""",""")))

    DEVICE = """cuda""" if torch.cuda.is_available() else """cpu"""
    print(f"""Using device: {DEVICE}""")

    envs = [SliceSimEnv(df, gnb_id=i) for i in cid_list]
    agents = [
        RLAgent(env.state_size, env.action_size, lr=HP.LR, device=DEVICE)
        for env in envs
    ]

    history = []
    for rd in trange(args.rounds, desc="""Fed Round"""):
        client_model_states = [ag.model.state_dict() for ag in agents]

        # 這裡假設您已修正 fedavg
        global_model_state = fedavg(client_model_states)

        for ag in agents:
            ag.model.load_state_dict(global_model_state)

        round_rewards = []
        round_losses = []

        for ag, env in zip(agents, envs):
            state, _ = env.reset()
            ep_reward = 0.0

            # 確保環境有足夠的數據
            if not env.ts:
                continue

            for step in range(HP.LOCAL_STEPS):
                act = ag.act(state)
                nst, rwd, term, trunc, _ = env.step(act)
                ag.remember(state, act, rwd, nst, term or trunc)

                loss = ag.replay(64)
                if loss is not None:
                    round_losses.append(loss)

                ep_reward += rwd
                state = nst
                if term or trunc:
                    break  # 如果環境結束，提前終止

            if HP.LOCAL_STEPS > 0:
                round_rewards.append(ep_reward / (step + 1))  # 使用實際步數計算平均獎勵

        avg_reward = float(np.mean(round_rewards)) if round_rewards else 0.0
        avg_loss = float(np.mean(round_losses)) if round_losses else 0.0

        history.append(
            {
                """round""": rd,
                """reward""": avg_reward,
                """loss""": avg_loss,
                """epsilon""": float(np.mean([ag.epsilon for ag in agents])),
            }
        )
        print(
            f"""Round {rd+1}/{args.rounds} - Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}"""
        )

    # 儲存歷史紀錄和模型
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / """training_history.csv""", index=False)
    torch.save(agents[0].model.state_dict(), out_dir / """global_model.pt""")

    # 繪製結果
    if not history_df.empty:
        plot_training_results(history_df["""reward"""], history_df["""loss"""])

    with open(out_dir / """config.json""", """w""") as fp:
        config_data = vars(args)
        # 將 Path 物件轉為字串以便 JSON 序列化
        config_data["""HP"""] = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in HP.__dict__.items()
            if not k.startswith("""__""")
        }
        config_data["""out"""] = str(config_data["""out"""])
        config_data["""parquet"""] = str(config_data["""parquet"""])
        json.dump(config_data, fp, indent=2)

    print(f"""✅  Finished {args.rounds} rounds – artifacts saved to {out_dir.resolve()}""")


# 修正：增加 if __name__ == """__main__""": 區塊
if __name__ == """__main__""":
    main()
