"""Quick test: verify modified reward function works correctly"""
import numpy as np
from envs.env_continuous import ContinuousActionEnv

env = ContinuousActionEnv()

print("=== Test 1: Random actions ===")
obs = env.reset()
total_rewards = []
for step in range(20):
    actions = [np.random.randn(a.shape[0]) for a in env.action_space]
    next_obs, rewards, dones, infos = env.step(actions)
    total_rewards.append(rewards)
    if step < 3:
        ud = infos[0]['reward_details']
        vd = infos[10]['reward_details']
        print(f'Step {step}: mean_rew={rewards.mean():.4f}, vio_rate={infos[0]["violation_rate"]:.3f}')
        print(f'  User0: total={ud["total"]:.4f}, sys={ud["system_reward"]:.4f}, w1={ud["w1_system"]:.4f}, w2={ud["w2_improvement"]:.4f}')
        print(f'  UAV0: total={vd["total"]:.4f}, guide={vd["centroid_guide"]:.4f}, coverage={vd.get("coverage_count",0)}, density={vd["density_guide"]:.4f}')

total_rewards = np.array(total_rewards)
print(f'Random: mean={total_rewards.mean():.4f}, std={total_rewards.std():.4f}')

print("\n=== Test 2: All local ===")
obs = env.reset()
rewards_local = []
for step in range(20):
    actions = []
    for i in range(10):
        act = np.zeros(5)
        act[0] = -5.0; act[1] = 5.0
        actions.append(act)
    for j in range(3):
        act = np.zeros(12); act[0] = -3.0
        actions.append(act)
    next_obs, rewards, dones, infos = env.step(actions)
    rewards_local.append(rewards)
    if step < 3:
        print(f'Step {step}: mean_rew={rewards.mean():.4f}, vio_rate={infos[0]["violation_rate"]:.3f}, cost_imp={infos[0]["cost_improvement"]:.4f}')

rewards_local = np.array(rewards_local)
print(f'Local: mean={rewards_local.mean():.4f}')

print("\n=== Test 3: Full offload to UAV0 ===")
obs = env.reset()
rewards_off = []
for step in range(20):
    actions = []
    for i in range(10):
        act = np.zeros(5)
        act[0] = 5.0; act[1] = -5.0; act[2] = 5.0
        actions.append(act)
    for j in range(3):
        act = np.zeros(12); act[0] = -3.0
        actions.append(act)
    next_obs, rewards, dones, infos = env.step(actions)
    rewards_off.append(rewards)
    if step < 3:
        ud = infos[0]['reward_details']
        print(f'Step {step}: mean_rew={rewards.mean():.4f}, vio_rate={infos[0]["violation_rate"]:.3f}')
        print(f'  User0: total={ud["total"]:.4f}, sys={ud["system_reward"]:.4f}')

rewards_off = np.array(rewards_off)
print(f'Offload: mean={rewards_off.mean():.4f}')

print(f"\n=== Summary ===")
print(f"Random:  {total_rewards.mean():.4f}")
print(f"Local:   {rewards_local.mean():.4f}")
print(f"Offload: {rewards_off.mean():.4f}")
print("Reward spread (random): user_avg={:.4f}, uav_avg={:.4f}".format(
    total_rewards[:,:10,:].mean(), total_rewards[:,10:,:].mean()))
