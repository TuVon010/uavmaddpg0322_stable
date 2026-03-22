import argparse
import torch
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gym.spaces import Box
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# 引入你的环境与配置
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.env_continuous import ContinuousActionEnv
from envs.Base import Base

from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = torch.cuda.is_available()

# =================================================================
# 轨迹绘制辅助函数
# =================================================================
def save_trajectory(episode, ep_user_pos, ep_uav_pos, n_users, n_uavs, traj_dir):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 画用户轨迹
    user_colors = plt.cm.tab10(np.linspace(0, 1, n_users))
    for i in range(n_users):
        pos = np.array(ep_user_pos[i])
        ax.plot(pos[:, 0], pos[:, 1], '-', color=user_colors[i], alpha=0.4, linewidth=0.8)
        ax.plot(pos[0, 0], pos[0, 1], 'o', color=user_colors[i], markersize=4) # 起点
        ax.plot(pos[-1, 0], pos[-1, 1], 's', color=user_colors[i], markersize=4) # 终点

    # 画UAV轨迹
    uav_colors = ['red', 'blue', 'green', 'purple', 'orange']
    uav_markers = ['D', '^', 'v', '<', '>']
    for j in range(n_uavs):
        pos = np.array(ep_uav_pos[j])
        c = uav_colors[j % len(uav_colors)]
        m = uav_markers[j % len(uav_markers)]
        ax.plot(pos[:, 0], pos[:, 1], linestyle='-', marker='.', color=c,
                linewidth=1.5, markersize=6, alpha=0.8, label=f'UAV {j}')
        ax.plot(pos[0, 0], pos[0, 1], m, color=c, markersize=10) # 起点
        ax.plot(pos[-1, 0], pos[-1, 1], '*', color=c, markersize=14) # 终点

    ax.set_xlim(0, 1000) # 若场地不是1000x1000可按需修改
    ax.set_ylim(0, 1000)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Episode {episode} Trajectories (o=start, */s=end)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    path = os.path.join(traj_dir, f'episode_{episode}.png')
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)

# =================================================================
# 环境实例化
# =================================================================
def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = ContinuousActionEnv()
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

# =================================================================
# 训练主循环
# =================================================================
def run(config):
    # 配置目录
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if str(folder.name).startswith('run')]
        curr_run = 'run1' if len(exst_run_nums) == 0 else 'run%i' % (max(exst_run_nums) + 1)
        
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    traj_dir = run_dir / 'trajectories'
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
        
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed, config.discrete_action)
    
    # 获取基础参数用于智能体分类
    base_config = Base()
    n_users = base_config.n_users
    n_uavs = base_config.n_uavs

    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg, adversary_alg=config.adversary_alg,
                                  tau=config.tau, lr=config.lr, hidden_dim=config.hidden_dim)
                                  
    # 使用类名字符串替代 isinstance，兼容 gym 和 gymnasium
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if acsp.__class__.__name__ == "Box" else acsp.n for acsp in env.action_space])
    t = 0
    
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        obs = env.reset()
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        # 初始化本轮回合的位置记录器（用于画轨迹）
        ep_user_pos = [[] for _ in range(n_users)]
        ep_uav_pos = [[] for _ in range(n_uavs)]

        for et_i in range(config.episode_length):
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(maddpg.nagents)]
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            
            next_obs, rewards, dones, infos = env.step(actions)
            info_e0 = infos[0]

            # 记录用于画图的位置
            for i in range(n_users):
                ep_user_pos[i].append(info_e0[i]['position'].copy())
            for j in range(n_uavs):
                ep_uav_pos[j].append(info_e0[n_users + j]['position'].copy())

            # 写入 Buffer 并更新网络
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            
            if (len(replay_buffer) >= config.batch_size and (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA: maddpg.prep_training(device='gpu')
                else: maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')

        # ==========================================
        # 轮次结束后的数据记录与日志
        # ==========================================
        # 1. 保存当前轮次的轨迹图 (每20轮保存一次,减少IO)
        if ep_i % 20 == 0:
            save_trajectory(ep_i, ep_user_pos, ep_uav_pos, n_users, n_uavs, traj_dir)

        # 2. 从 Buffer 中获取最近一个 Episode 的平均每步奖励
        ep_step_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)

        # 将平均每步奖励乘以轮次长度，得到该轮次的【总奖励和】
        agent_ep_rewards = [rew * config.episode_length for rew in ep_step_rews]

        # 写入每个智能体各自的总奖励
        for a_i, a_ep_rew in enumerate(agent_ep_rewards):
            logger.add_scalar(f'agent{a_i}/episode_reward', a_ep_rew, ep_i)

        # 3. 计算所有智能体的平均轮次总奖励（系统整体水平），并写入 TensorBoard
        avg_episode_reward = np.mean(agent_ep_rewards)
        user_avg_reward = np.mean(agent_ep_rewards[:n_users])
        uav_avg_reward = np.mean(agent_ep_rewards[n_users:])
        logger.add_scalar('Metrics/avg_episode_reward_per_agent', avg_episode_reward, ep_i)
        logger.add_scalar('Metrics/user_avg_reward', user_avg_reward, ep_i)
        logger.add_scalar('Metrics/uav_avg_reward', uav_avg_reward, ep_i)

        # 4. 记录环境指标 (使用最后一步的info)
        if info_e0 is not None and len(info_e0) > 0:
            logger.add_scalar('Metrics/violation_rate', info_e0[0].get('violation_rate', 0), ep_i)
            logger.add_scalar('Metrics/cost_improvement', info_e0[0].get('cost_improvement', 0), ep_i)
            logger.add_scalar('Metrics/norm_weighted_cost', info_e0[0].get('norm_weighted_cost', 0), ep_i)

        # 5. 控制台打印进度
        if ep_i % 50 == 0:
            vio = info_e0[0].get('violation_rate', -1) if info_e0 is not None else -1
            ci = info_e0[0].get('cost_improvement', 0) if info_e0 is not None else 0
            print(f"Ep {ep_i+1}/{config.n_episodes} | Avg R: {avg_episode_reward:.3f} (U:{user_avg_reward:.3f} D:{uav_avg_reward:.3f}) | Vio: {vio:.2f} | CI: {ci:.3f}")

        # 保存模型
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", default="UAV_Medical_IoT", nargs='?')
    parser.add_argument("model_name", default="MADDPG_Test", nargs='?')
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e5), type=int)
    parser.add_argument("--n_episodes", default=5000, type=int)
    parser.add_argument("--episode_length", default=60, type=int)
    parser.add_argument("--steps_per_update", default=10, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--n_exploration_eps", default=4000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.02, type=float)
    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--agent_alg", default="DDPG", type=str)
    parser.add_argument("--adversary_alg", default="DDPG", type=str)
    parser.add_argument("--discrete_action", action='store_true')

    config = parser.parse_args()
    run(config)