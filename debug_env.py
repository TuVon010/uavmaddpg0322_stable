"""Debug script to check environment reward signals"""
import numpy as np
from envs.env_continuous import ContinuousActionEnv

env = ContinuousActionEnv()
obs = env.reset()
print('Obs shape:', obs.shape)
print('Num agents:', env.num_agent)
print('Action dims:', [a.shape[0] for a in env.action_space])
print('Obs dims:', [o.shape[0] for o in env.observation_space])

# Run a few random steps to check reward magnitudes
total_rewards = []
for step in range(20):
    actions = [np.random.randn(a.shape[0]) for a in env.action_space]
    next_obs, rewards, dones, infos = env.step(actions)
    total_rewards.append(rewards)
    if step < 3:
        print(f'\nStep {step}: rewards range [{rewards.min():.4f}, {rewards.max():.4f}], mean={rewards.mean():.4f}')
        # Print reward details for first user and first UAV
        ud = infos[0]['reward_details']
        print(f'  User0: sys_rew={ud["system_reward"]:.4f}, w1_sys={ud["w1_system"]:.4f}, w2_imp={ud["w2_improvement"]:.4f}, total={ud["total"]:.4f}')
        print(f'  User0: delay_ratio={ud["delay_ratio"]:.4f}, energy_ratio={ud["energy_ratio"]:.4f}')
        vd = infos[10]['reward_details']
        print(f'  UAV0: sys_rew={vd["system_reward"]:.4f}, w_sys={vd["w_sys_part"]:.4f}, guide={vd["centroid_guide"]:.4f}, density={vd["density_guide"]:.4f}')
        print(f'  UAV0: delay_ben={vd["delay_benefit"]:.4f}, energy_pen={vd["energy_pen"]:.4f}, boundary={vd["boundary_pen"]:.4f}, collision={vd["collision_pen"]:.4f}')
        print(f'  UAV0: uav_individual={vd["uav_individual"]:.4f}, w_ind={vd["w_ind_part"]:.4f}, total={vd["total"]:.4f}')
        print(f'  Violation rate: {infos[0]["violation_rate"]:.3f}')
        print(f'  Cost improvement: {infos[0]["cost_improvement"]:.4f}')
        print(f'  Norm weighted cost: {infos[0]["norm_weighted_cost"]:.4f}')
        print(f'  Norm weighted cost base: {infos[0]["norm_weighted_cost_base"]:.4f}')

total_rewards = np.array(total_rewards)
print(f'\nOverall: mean={total_rewards.mean():.4f}, std={total_rewards.std():.4f}')
print(f'Min={total_rewards.min():.4f}, Max={total_rewards.max():.4f}')

# Check per-agent type averages
user_rewards = total_rewards[:, :10, :].mean()
uav_rewards = total_rewards[:, 10:, :].mean()
print(f'User avg reward: {user_rewards:.4f}')
print(f'UAV avg reward: {uav_rewards:.4f}')

# Check what happens with "smart" actions - all local compute
print('\n--- Testing all-local compute (ratio=0, assoc=local) ---')
obs = env.reset()
for step in range(5):
    actions = []
    for i in range(10):  # users: ratio=0 (local), assoc=local (first logit high)
        act = np.zeros(5)
        act[0] = -5.0  # sigmoid(-5) ~ 0 -> all local
        act[1] = 5.0   # high logit for local
        actions.append(act)
    for j in range(3):  # UAVs: low speed, some direction
        act = np.zeros(12)
        act[0] = -3.0  # low speed
        actions.append(act)
    next_obs, rewards, dones, infos = env.step(actions)
    if step < 3:
        ud = infos[0]['reward_details']
        print(f'Step {step}: mean_rew={rewards.mean():.4f}, violation_rate={infos[0]["violation_rate"]:.3f}, cost_imp={infos[0]["cost_improvement"]:.4f}')
        print(f'  User0: total={ud["total"]:.4f}, sys_rew={ud["system_reward"]:.4f}')

# Check what happens with offloading
print('\n--- Testing offloading (ratio=1, assoc=UAV0) ---')
obs = env.reset()
for step in range(5):
    actions = []
    for i in range(10):  # users: full offload to UAV 0
        act = np.zeros(5)
        act[0] = 5.0   # sigmoid(5) ~ 1 -> full offload
        act[1] = -5.0  # low logit for local
        act[2] = 5.0   # high logit for UAV 0
        actions.append(act)
    for j in range(3):  # UAVs: move toward center
        act = np.zeros(12)
        act[0] = -3.0  # low speed
        # equal freq allocation
        act[2:] = 0.0
        actions.append(act)
    next_obs, rewards, dones, infos = env.step(actions)
    if step < 3:
        ud = infos[0]['reward_details']
        print(f'Step {step}: mean_rew={rewards.mean():.4f}, violation_rate={infos[0]["violation_rate"]:.3f}, cost_imp={infos[0]["cost_improvement"]:.4f}')
        print(f'  User0: total={ud["total"]:.4f}, sys_rew={ud["system_reward"]:.4f}, delay_ratio={ud["delay_ratio"]:.4f}')
        print(f'  Norm cost={infos[0]["norm_weighted_cost"]:.4f}, base={infos[0]["norm_weighted_cost_base"]:.4f}')
