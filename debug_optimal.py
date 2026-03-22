"""Test: verify offloading can actually beat local under optimal conditions"""
import numpy as np
from envs.env_continuous import ContinuousActionEnv
from envs.Base import Base

b = Base()
env = ContinuousActionEnv()
obs = env.reset()

# Move UAVs directly to center
for j in range(3):
    env.env.uavs[j]['position'] = np.array([500.0 + (j-1)*80, 500.0])

user_center = np.mean([u['position'] for u in env.env.users], axis=0)
for j in range(3):
    d = np.linalg.norm(env.env.uavs[j]['position'] - user_center)
    print(f"UAV {j} at {env.env.uavs[j]['position']}, dist to centroid: {d:.0f}m")

print("\n=== Test with forced UAV positions (near users) ===")
# Test 1: Only offload big tasks to nearest UAV
print("\n--- Strategy A: Only offload tasks > 1.5MB, balanced across UAVs ---")
rewards_A = []
for step in range(30):
    actions = []
    # Assign each user to nearest UAV
    for i in range(10):
        act = np.zeros(5)
        D = env.env.tasks[i]['data_size']
        C = env.env.tasks[i]['cpu_cycles']
        T_local = D * C / b.C_local

        # Find nearest UAV
        user_pos = env.env.users[i]['position']
        uav_dists = [np.linalg.norm(user_pos - env.env.uavs[j]['position']) for j in range(3)]
        nearest_uav = int(np.argmin(uav_dists))
        nearest_dist = uav_dists[nearest_uav]

        # Only offload if: task is big enough AND UAV is close enough
        if T_local > 0.3 and nearest_dist < b.coverage_radius:
            # Partial offload: bigger task -> higher ratio
            offload_ratio = min(0.8, T_local / b.latency_max)
            act[0] = np.log(offload_ratio / (1 - offload_ratio + 1e-8))  # inverse sigmoid
            act[1] = -5.0  # not local
            act[2 + nearest_uav] = 5.0
        else:
            act[0] = -5.0  # all local
            act[1] = 5.0
        actions.append(act)

    for j in range(3):
        act = np.zeros(12)
        act[0] = -5.0  # stay still
        act[2:] = 0.0  # equal freq
        actions.append(act)

    next_obs, rewards, dones, infos = env.step(actions)
    rewards_A.append(rewards)
    if step < 5:
        vr = infos[0]['violation_rate']
        ci = infos[0]['cost_improvement']
        di = infos[0]['reward_details']['delay_saving'] if 'delay_saving' in infos[0]['reward_details'] else 0
        print(f"  Step {step}: mean={rewards.mean():.4f}, vio={vr:.2f}, cost_imp={ci:.4f}, delay_imp={di:.4f}")
        # Count offloading users
        n_off = sum(1 for inf in infos[:10] if inf.get('offload_ratio', 0) > 0.1)
        print(f"    Offloading users: {n_off}/10")

    # Reset UAV positions (prevent drifting)
    for j in range(3):
        env.env.uavs[j]['position'] = np.array([500.0 + (j-1)*80, 500.0])

rewards_A = np.array(rewards_A)
print(f"Strategy A: mean={rewards_A.mean():.4f}")

# Test 2: All local baseline
print("\n--- Strategy B: All local ---")
obs = env.reset()
for j in range(3):
    env.env.uavs[j]['position'] = np.array([500.0 + (j-1)*80, 500.0])

rewards_B = []
for step in range(30):
    actions = []
    for i in range(10):
        act = np.zeros(5); act[0] = -5.0; act[1] = 5.0
        actions.append(act)
    for j in range(3):
        act = np.zeros(12); act[0] = -5.0
        actions.append(act)
    next_obs, rewards, dones, infos = env.step(actions)
    rewards_B.append(rewards)
    for j in range(3):
        env.env.uavs[j]['position'] = np.array([500.0 + (j-1)*80, 500.0])

rewards_B = np.array(rewards_B)
print(f"Strategy B: mean={rewards_B.mean():.4f}")

print(f"\n=== RESULT ===")
print(f"Smart offload (A): {rewards_A.mean():.4f}")
print(f"All local (B):     {rewards_B.mean():.4f}")
print(f"Difference: {rewards_A.mean() - rewards_B.mean():.4f} ({'offload wins!' if rewards_A.mean() > rewards_B.mean() else 'local still wins'})")
