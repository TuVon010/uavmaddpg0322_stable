"""Test: simulate smart play where UAV flies to users first, then offload"""
import numpy as np
from envs.env_continuous import ContinuousActionEnv

env = ContinuousActionEnv()

print("=== Simulate: UAV flies to center, then users offload ===")
obs = env.reset()

# Print initial positions
for j in range(3):
    print(f"UAV {j} initial pos: {env.env.uavs[j]['position']}")
user_center = np.mean([u['position'] for u in env.env.users], axis=0)
print(f"User centroid: {user_center}")

# Phase 1: UAVs fly toward center at max speed for 25 steps
# Users do local compute during this phase
print("\n--- Phase 1: UAVs flying to center (25 steps) ---")
for step in range(25):
    actions = []
    for i in range(10):
        act = np.zeros(5)
        act[0] = -5.0; act[1] = 5.0  # all local
        actions.append(act)
    for j in range(3):
        act = np.zeros(12)
        # Calculate direction toward user centroid
        uav_pos = env.env.uavs[j]['position']
        target = user_center
        vec = target - uav_pos
        angle = np.arctan2(vec[1], vec[0])
        # act[0] = sigmoid(x) * v_max, so x=5 -> speed ~ v_max
        act[0] = 5.0  # max speed
        # act[1] = tanh(x) * pi, need to set x such that tanh(x)*pi = angle
        act[1] = np.arctanh(np.clip(angle / np.pi, -0.99, 0.99))
        actions.append(act)

    next_obs, rewards, dones, infos = env.step(actions)
    if step % 5 == 0:
        dists = [np.linalg.norm(env.env.uavs[j]['position'] - user_center) for j in range(3)]
        print(f"  Step {step}: UAV dists to center: [{dists[0]:.0f}, {dists[1]:.0f}, {dists[2]:.0f}]m, mean_rew={rewards.mean():.4f}")

# Print UAV positions after flying
for j in range(3):
    d = np.linalg.norm(env.env.uavs[j]['position'] - user_center)
    print(f"  UAV {j} pos: {env.env.uavs[j]['position']}, dist to center: {d:.0f}m")

# Phase 2: Smart offloading - distribute users across UAVs
print("\n--- Phase 2: Smart offloading (UAVs near users) ---")
rewards_phase2 = []
for step in range(25, 50):
    actions = []
    for i in range(10):
        act = np.zeros(5)
        # Assign users to UAVs: 0-2->UAV0, 3-5->UAV1, 6-9->UAV2
        target_uav = i // 4 if i < 9 else 2  # roughly balanced
        # Check distance to assigned UAV
        d_to_uav = np.linalg.norm(env.env.users[i]['position'] - env.env.uavs[target_uav]['position'])
        if d_to_uav < 200:  # within reasonable range
            act[0] = 3.0   # sigmoid(3) ~ 0.95 -> mostly offload
            act[1] = -5.0  # low logit for local
            act[2 + target_uav] = 5.0  # high logit for target UAV
        else:
            act[0] = -5.0; act[1] = 5.0  # too far, stay local
        actions.append(act)
    for j in range(3):
        act = np.zeros(12)
        act[0] = -3.0  # slow speed (already near target)
        # Equal freq allocation
        act[2:] = 0.0
        actions.append(act)

    next_obs, rewards, dones, infos = env.step(actions)
    rewards_phase2.append(rewards)
    if step % 5 == 0:
        vio_rate = infos[0]['violation_rate']
        cost_imp = infos[0]['cost_improvement']
        norm_cost = infos[0]['norm_weighted_cost']
        norm_base = infos[0]['norm_weighted_cost_base']
        ud = infos[0]['reward_details']
        print(f"  Step {step}: mean_rew={rewards.mean():.4f}, vio_rate={vio_rate:.3f}, cost_imp={cost_imp:.4f}")
        print(f"    norm_cost={norm_cost:.4f} vs base={norm_base:.4f}")
        print(f"    User0: total={ud['total']:.4f}, sys={ud['system_reward']:.4f}, w2_imp={ud['w2_improvement']:.4f}")

rewards_phase2 = np.array(rewards_phase2)
print(f"\nPhase 2 (smart offload): mean={rewards_phase2.mean():.4f}")
print(f"  User avg: {rewards_phase2[:,:10,:].mean():.4f}")
print(f"  UAV avg: {rewards_phase2[:,10:,:].mean():.4f}")

# Phase 3: Compare - all local in phase 2 (no offload baseline)
print("\n--- Phase 3: All local during phase 2 for comparison ---")
obs = env.reset()
for step in range(25):
    actions = []
    for i in range(10):
        act = np.zeros(5); act[0] = -5.0; act[1] = 5.0
        actions.append(act)
    for j in range(3):
        act = np.zeros(12)
        uav_pos = env.env.uavs[j]['position']
        vec = user_center - uav_pos
        angle = np.arctan2(vec[1], vec[0])
        act[0] = 5.0
        act[1] = np.arctanh(np.clip(angle / np.pi, -0.99, 0.99))
        actions.append(act)
    next_obs, rewards, dones, infos = env.step(actions)

rewards_local_phase2 = []
for step in range(25, 50):
    actions = []
    for i in range(10):
        act = np.zeros(5); act[0] = -5.0; act[1] = 5.0
        actions.append(act)
    for j in range(3):
        act = np.zeros(12); act[0] = -3.0
        actions.append(act)
    next_obs, rewards, dones, infos = env.step(actions)
    rewards_local_phase2.append(rewards)

rewards_local_phase2 = np.array(rewards_local_phase2)
print(f"All-local in phase 2: mean={rewards_local_phase2.mean():.4f}")

print(f"\n=== COMPARISON (Phase 2 only) ===")
print(f"Smart offload: {rewards_phase2.mean():.4f}")
print(f"All local:     {rewards_local_phase2.mean():.4f}")
print(f"Difference:    {rewards_phase2.mean() - rewards_local_phase2.mean():.4f} ({'offload better' if rewards_phase2.mean() > rewards_local_phase2.mean() else 'local better'})")
