# ─── scripts/train.py ───
from __future__ import annotations
import argparse, json, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import trange

from colosseum_oran_frl_demo.config   import HP, Paths
from colosseum_oran_frl_demo.envs.slice_sim_env import SliceSimEnv
from colosseum_oran_frl_demo.agents.rl_agent      import RLAgent
from colosseum_oran_frl_demo.agents.fed_server    import fedavg

# ---------- CLI ---------------------------------------------------------- #
ap = argparse.ArgumentParser(description="Offline FRL training entrypoint")
ap.add_argument("--parquet", default=Paths.PROCESSED, help="Parquet root or file")
ap.add_argument("--rounds",  type=int, default=5)
ap.add_argument("--clients", default="1,2,3")
ap.add_argument("--out",     default=Paths.OUTPUTS)
args = ap.parse_args()

out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
df      = (Path(args.parquet).is_dir() and
           pd.read_parquet(list(Path(args.parquet).glob("*.parquet"))[0])) \
          or pd.read_parquet(args.parquet)
cid_list = list(map(int, args.clients.split(",")))

envs   = [SliceSimEnv(df, gnb_id=i) for i in cid_list]
agents = [RLAgent(env.state_size, env.action_size,
                  lr=HP.LR, device="cpu") for env in envs]

history = []
for rd in trange(args.rounds, desc="Fed Round"):
    round_rewards = []
    for ag, env in zip(agents, envs):
        state,_ = env.reset()
        ep_reward = 0.0
        for _ in range(HP.LOCAL_STEPS):
            act  = ag.act(state)
            nst, rwd, term, trunc, _ = env.step(act)
            ag.remember(state, act, rwd, nst, term or trunc)
            ag.replay(64)
            ep_reward += rwd
            state = nst
        round_rewards.append(ep_reward / HP.LOCAL_STEPS)
    fedavg(agents)
    history.append({
        "round": rd,
        "reward": float(np.mean(round_rewards)),
        "epsilon": float(np.mean([ag.epsilon for ag in agents]))
    })

pd.DataFrame(history).to_csv(out_dir/"training_history.csv", index=False)
torch.save(agents[0].model.state_dict(), out_dir/"global_model.pt")
with open(out_dir/"config.json", "w") as fp:
    json.dump(vars(args)|{"HP": HP.__dict__}, fp, indent=2)

print(f"✅  Finished {args.rounds} rounds – artifacts saved to {out_dir.resolve()}")
