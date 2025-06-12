from __future__ import annotations
import argparse, json, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import trange

from colosseum_oran_frl_demo.config import HP, Paths
from colosseum_oran_frl_demo.envs.slice_sim_env import SliceSimEnv
from colosseum_oran_frl_demo.agents.rl_agent import RLAgent
from colosseum_oran_frl_demo.agents.fed_server import fedavg
from colosseum_oran_frl_demo.utils.plots import plot_training_results # 修正：匯入繪圖函式

# ---------- CLI ---------------------------------------------------------- #
ap = argparse.ArgumentParser(description="Offline FRL training entrypoint")
ap.add_argument("--parquet", default=str(Paths.PROCESSED / "kpi_traces_final_v_robust.parquet"), help="Path to the final processed Parquet file")
ap.add_argument("--rounds", type=int, default=10)
ap.add_argument("--clients", default="1,2,3")
ap.add_argument("--out", default=str(Paths.OUTPUTS))
args = ap.parse_args()

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(args.parquet)
cid_list = list(map(int, args.clients.split(",")))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

envs = [SliceSimEnv(df, gnb_id=i) for i in cid_list]
agents = [
    RLAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        lr=HP.LR,
        device=DEVICE,
    )
    for env in envs
]

# 修正：使用一個全域代理來持有聚合後的模型
global_agent = RLAgent(envs[0].state_size, envs[0].action_size, device=DEVICE)

history = []
for rd in trange(args.rounds, desc="Fed Round"):
    client_models = [ag.model.state_dict() for ag in agents]
    
    # 修正：從伺服器聚合模型
    global_model_state = fedavg(client_models)
    
    # 修正：將聚合後的模型分發給所有客戶端
    for ag in agents:
        ag.model.load_state_dict(global_model_state)

    round_rewards = []
    round_losses = []
    
    for ag, env in zip(agents, envs):
        if not env.is_ready(): # 檢查環境是否有足夠數據
            continue
            
        state, _ = env.reset()
        ep_reward = 0.0
        
        for _ in range(HP.LOCAL_STEPS):
            act = ag.act(state)
            nst, rwd, term, trunc, _ = env.step(act)
            ag.remember(state, act, rwd, nst, term or trunc)
            
            # 修正：收集 replay 的 loss
            loss = ag.replay(64)
            if loss is not None:
                round_losses.append(loss)
                
            ep_reward += rwd
            state = nst
            
            if term or trunc:
                break
                
        round_rewards.append(ep_reward / HP.LOCAL_STEPS)

    avg_reward = float(np.mean(round_rewards)) if round_rewards else 0.0
    avg_loss = float(np.mean(round_losses)) if round_losses else 0.0
    
    history.append({
        "round": rd,
        "reward": avg_reward,
        "loss": avg_loss, # 修正：增加 loss 記錄
        "epsilon": float(np.mean([ag.epsilon for ag in agents])),
    })
    
    print(f'Round {rd+1}/{args.rounds} - Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}')

# 修正：儲存最終的全域模型
torch.save(global_agent.model.state_dict(), out_dir / "global_model.pt")

# 修正：儲存訓練歷史
history_df = pd.DataFrame(history)
history_df.to_csv(out_dir / "training_history.csv", index=False)

# 修正：呼叫繪圖函式
plot_training_results(history_df['reward'], history_df['loss'])

with open(out_dir / "config.json", "w") as fp:
    # 修正：儲存超參數
    config_data = vars(args)
    config_data['HP'] = HP.__dict__
    json.dump(config_data, fp, indent=2, default=str) # 處理 Path 物件

print(f"✅ Finished {args.rounds} rounds – artifacts saved to {out_dir.resolve()}")
