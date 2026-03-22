"""Evaluate trained model from run2"""
import torch
import numpy as np
from envs.env_continuous import ContinuousActionEnv
from envs.Base import Base
from algorithms.maddpg import MADDPG

# Load trained model
model_path = './models/UAV_Medical_IoT/MADDPG_V3/run2/model.pt'
maddpg = MADDPG.init_from_save(model_path)
maddpg.prep_rollouts(device='cpu')

env = ContinuousActionEnv()
base = Base()
n_users = base.n_users
n_uavs = base.n_uavs

# Evaluate over multiple episodes (no exploration noise)
n_eval = 20
all_rewards = []
all_violations = []
all_cost_imp = []
all_delay_imp = []

for ep in range(n_eval):
    obs = env.reset()
    ep_rewards = []
    last_info = None
    for step in range(60):
        torch_obs = [torch.Tensor(np.expand_dims(obs[i], 0)) for i in range(maddpg.nagents)]
        actions = [a.step(o, explore=False) for a, o in zip(maddpg.agents, torch_obs)]
        actions_np = [ac.data.numpy().flatten() for ac in actions]

        next_obs, rewards, dones, infos = env.step(actions_np)
        ep_rewards.append(rewards)
        last_info = infos
        obs = next_obs

    ep_rewards = np.array(ep_rewards)
    ep_total = ep_rewards.sum(axis=0).mean()
    all_rewards.append(ep_total)

    vio = last_info[0].get('violation_rate', -1) if last_info else -1
    ci = last_info[0].get('cost_improvement', 0) if last_info else 0
    di = last_info[0]['reward_details'].get('delay_saving', 0) if last_info else 0
    all_violations.append(vio)
    all_cost_imp.append(ci)
    all_delay_imp.append(di)

    if ep < 5:
        print(f"Ep {ep}: total_reward={ep_total:.3f}, vio_rate={vio:.3f}, cost_imp={ci:.4f}, delay_imp={di:.4f}")
        print(f"  User avg: {ep_rewards[:,:n_users,:].sum(axis=0).mean():.3f}")
        print(f"  UAV avg: {ep_rewards[:,n_users:,:].sum(axis=0).mean():.3f}")

print(f"\n=== {n_eval} Episode Evaluation ===")
print(f"Avg Total Reward: {np.mean(all_rewards):.3f} +/- {np.std(all_rewards):.3f}")
print(f"Avg Violation Rate: {np.mean(all_violations):.3f}")
print(f"Avg Cost Improvement: {np.mean(all_cost_imp):.4f}")
print(f"Avg Delay Improvement: {np.mean(all_delay_imp):.4f}")

# Compare with random and local baselines
print("\n=== Baselines ===")
random_rewards = []
local_rewards = []
for ep in range(n_eval):
    obs = env.reset()
    ep_r_rand = []
    ep_r_local = []

    obs2 = env.reset()
    for step in range(60):
        # Random
        rand_actions = [np.random.randn(a.shape[0]) for a in env.action_space]
        _, r_rand, _, _ = env.step(rand_actions)
        ep_r_rand.append(r_rand)

    obs = env.reset()
    for step in range(60):
        # All local
        local_actions = []
        for i in range(n_users):
            act = np.zeros(5); act[0] = -5.0; act[1] = 5.0
            local_actions.append(act)
        for j in range(n_uavs):
            act = np.zeros(12); act[0] = -5.0
            local_actions.append(act)
        _, r_local, _, _ = env.step(local_actions)
        ep_r_local.append(r_local)

    random_rewards.append(np.array(ep_r_rand).sum(axis=0).mean())
    local_rewards.append(np.array(ep_r_local).sum(axis=0).mean())

print(f"Random baseline: {np.mean(random_rewards):.3f}")
print(f"All-local baseline: {np.mean(local_rewards):.3f}")
print(f"Trained model: {np.mean(all_rewards):.3f}")
improvement_vs_random = (np.mean(all_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100
improvement_vs_local = (np.mean(all_rewards) - np.mean(local_rewards)) / abs(np.mean(local_rewards)) * 100
print(f"Improvement vs random: {improvement_vs_random:+.1f}%")
print(f"Improvement vs local: {improvement_vs_local:+.1f}%")
