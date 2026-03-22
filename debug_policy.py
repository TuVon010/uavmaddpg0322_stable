"""Detailed evaluation: check what the trained policy actually does"""
import torch
import numpy as np
from envs.env_continuous import ContinuousActionEnv
from envs.Base import Base
from algorithms.maddpg import MADDPG

model_path = './models/UAV_Medical_IoT/MADDPG_V3/run1/model.pt'
maddpg = MADDPG.init_from_save(model_path)
maddpg.prep_rollouts(device='cpu')

env = ContinuousActionEnv()
base = Base()

obs = env.reset()
user_center = np.mean([u['position'] for u in env.env.users], axis=0)
print(f"User centroid: {user_center}")

for step in range(60):
    torch_obs = [torch.Tensor(np.expand_dims(obs[i], 0)) for i in range(maddpg.nagents)]
    actions = [a.step(o, explore=False) for a, o in zip(maddpg.agents, torch_obs)]
    actions_np = [ac.data.numpy().flatten() for ac in actions]

    if step % 10 == 0:
        print(f"\n--- Step {step} ---")
        # Check UAV positions and distances
        for j in range(3):
            pos = env.env.uavs[j]['position']
            d = np.linalg.norm(pos - user_center)
            raw_act = actions_np[10+j]
            from envs.env_core import _sigmoid
            speed = _sigmoid(raw_act[0]) * base.uav_v_max
            direction = np.tanh(raw_act[1]) * np.pi
            print(f"  UAV{j}: pos=({pos[0]:.0f},{pos[1]:.0f}), dist_to_center={d:.0f}m, speed={speed:.1f}m/s, dir={np.degrees(direction):.0f}deg")

        # Check user actions
        n_offload = 0
        for i in range(10):
            raw_act = actions_np[i]
            ratio = _sigmoid(raw_act[0])
            assoc = int(np.argmax(raw_act[1:5]))
            if assoc > 0:
                n_offload += 1
            if i < 3:
                uav_dists = [np.linalg.norm(env.env.users[i]['position'] - env.env.uavs[j]['position']) for j in range(3)]
                min_d = min(uav_dists)
                print(f"  User{i}: ratio={ratio:.3f}, assoc={assoc} {'(local)' if assoc==0 else f'(UAV{assoc-1})'}, min_dist_to_uav={min_d:.0f}m")
        print(f"  Offloading users: {n_offload}/10")

    next_obs, rewards, dones, infos = env.step(actions_np)
    obs = next_obs

    if step % 10 == 0:
        vr = infos[0]['violation_rate']
        ci = infos[0]['cost_improvement']
        print(f"  Reward: mean={rewards.mean():.4f}, vio={vr:.2f}, cost_imp={ci:.4f}")
