import time
import os
import csv
import numpy as np
from itertools import chain
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner
import pandas as pd # <=== [新增] 必须导入 pandas
from envs.Base import Base  # 直接引入你的 Base 配置
'''_collect_performance_metrics这里面有几个数据是累加的，带cum的'''
def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

        env_core = self.envs.envs[0].env
        self.n_users = env_core.n_users
        self.n_uavs = env_core.n_uavs
        # [新增] 实例化 Base 以便获取各项权重
        self.Base = Base()

        if not self.use_render:
            self.traj_dir = str(self.run_dir / 'trajectories')
            self.reward_dir = str(self.run_dir / 'reward_details')
            # [新增] 创建专门存放时延能耗等性能指标的目录
            self.metrics_dir = str(self.run_dir / 'performance_metrics') 
            os.makedirs(self.traj_dir, exist_ok=True)
            os.makedirs(self.reward_dir, exist_ok=True)
            os.makedirs(self.metrics_dir, exist_ok=True) # [新增]
    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].set_progress(episode, episodes)

            # 基本位置与系统指标记录
            ep_user_pos = [[] for _ in range(self.n_users)]
            ep_uav_pos = [[] for _ in range(self.n_uavs)]
            ep_delays = []
            ep_sys_energies = []
            ep_sys_time_costs = []
            ep_uav_fly_energies = []
            ep_norm_weighted_costs = []
            ep_norm_weighted_cost_bases = []
            
            # ==========================================
            # 1. 初始化【奖励详情表】的容器和累计变量
            # ==========================================
            ep_reward_rows = []
            sys_cum_reward = 0.0 # 系统的累计大锅饭奖励
            agent_cum_rewards = np.zeros(self.num_agents) # 每个智能体的累计总分
            
            # ==========================================
            # 2. 初始化【3张性能指标表】的容器和累计变量
            # ==========================================
            ep_user_metrics = []
            ep_uav_metrics = []
            ep_sys_metrics = []
            user_cum_energy = np.zeros(self.n_users)
            sys_cum_time_cost = 0.0
            sys_cum_energy = 0.0
            

            for step in range(self.episode_length):
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                obs, rewards, dones, infos = self.envs.step(actions_env)

                # --- 从第一个环境线程收集指标 ---
                info_e0 = infos[0]
                
                # 记录位置轨迹和基础指标
                for i in range(self.n_users):
                    ep_user_pos[i].append(info_e0[i]['position'].copy())
                for j in range(self.n_uavs):
                    ep_uav_pos[j].append(info_e0[self.n_users + j]['position'].copy())
                ep_delays.append(info_e0[0]['avg_user_delay'])
                ep_sys_energies.append(info_e0[0]['total_system_energy'])
                ep_sys_time_costs.append(info_e0[0]['sys_time_cost'])
                ep_norm_weighted_costs.append(info_e0[0].get('norm_weighted_cost', 0.0))
                ep_norm_weighted_cost_bases.append(info_e0[0].get('norm_weighted_cost_base', 0.0))
                uav_fly_e_step = sum(info_e0[self.n_users + j]['fly_energy']
                                     for j in range(self.n_uavs))
                ep_uav_fly_energies.append(uav_fly_e_step)

                # ========================================================
                # [追加新增] 提取并保存3张表需要的性能数据 (已封装)
                # ========================================================
                if episode % 10 == 0:
                    sys_cum_time_cost, sys_cum_energy = self._collect_performance_metrics(
                        step, info_e0,
                        ep_sys_metrics, ep_user_metrics, ep_uav_metrics,
                        sys_cum_time_cost, sys_cum_energy, user_cum_energy
                    )

                # ========================================================
                # 记录 B：提取并保存高度拆解的【奖励详情表】数据
                # ========================================================
                if episode % 10 == 0:
                    # 1. 提取本步的系统大锅饭奖励 (全局共享，直接取第0个就行)
                    step_sys_reward = info_e0[0]['reward_details']['system_reward']
                    sys_cum_reward += step_sys_reward

                    # 2. 遍历每个智能体，分类提取数据
                    for a in range(self.num_agents):
                        rd = info_e0[a]['reward_details']
                        step_total = rd['total']
                        agent_cum_rewards[a] += step_total
                        
                        # 生成当前步的报表行 (利用 .get() 方法，没有的项自动补 0.0)
                        row = {
                            'Step': step,
                            'Agent_Type': rd['agent_type'].upper(), # 'USER' 或 'UAV'
                            'Agent_ID': a if a < self.n_users else a - self.n_users,
                            
                            # --- 全局共有项 ---
                            'System_Reward': step_sys_reward,
                            'System_Cumulative_Reward': sys_cum_reward,
                            'Agent_Step_Total_Reward': step_total,
                            'Agent_Cumulative_Reward': agent_cum_rewards[a],
                            'System_Cost_Improve': info_e0[0].get('cost_improvement', 0.0),
                            'System_Reward_Raw': info_e0[0].get('system_reward_raw', 0.0),
                            
                            # --- 用户专属字段 (UAV填0) ---
                            'w1_System_Reward': rd.get('w1_system', 0.0),
                            'w2_Cost_Improvement': rd.get('w2_improvement', 0.0),
                            'Norm_Delay_Ratio': rd.get('delay_ratio', 0.0),
                            'Norm_Energy_Ratio': rd.get('energy_ratio', 0.0),
                            
                            # --- 无人机专属字段 (User填0) ---
                            'w_Sys_Part_Reward': rd.get('w_sys_part', 0.0),
                            'Centroid_Guide': rd.get('centroid_guide', 0.0),
                            'Density_Guide': rd.get('density_guide', 0.0),
                            'Delay_Benefit': rd.get('delay_benefit', 0.0),
                            'Boundary_Penalty': rd.get('boundary_pen', 0.0),
                            'UAV_Energy_Penalty': rd.get('energy_pen', 0.0), # <=== 接住无人机能耗惩罚
                            'Collision_Penalty': rd.get('collision_pen', 0.0),
                            'Guide_Mode': rd.get('guide_mode', ''),
                            'Assoc_Mix': rd.get('assoc_mix', 0.0),
                        }
                        ep_reward_rows.append(row)
                

                data = (
                    obs, rewards, dones, infos, values, actions,
                    action_log_probs, rnn_states, rnn_states_critic,
                )
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # --- 每个轮次: 保存轨迹图 ---
            self._save_trajectory(episode, ep_user_pos, ep_uav_pos)

            # --- 每5个轮次: 打印平均时延和平均能耗 ---
            if episode % 5 == 0:
                self._print_episode_metrics(episode, ep_delays, ep_sys_energies,
                                            ep_sys_time_costs, ep_uav_fly_energies,
                                            ep_norm_weighted_costs)

            # --- 每10个轮次: 保存奖励详情表格 ---
            if episode % 10 == 0:
                self._save_reward_table(episode, ep_reward_rows)
                # [追加新增] 顺便把新的 3 张性能指标表也存下来
                self._save_performance_metrics(episode, ep_user_metrics, ep_uav_metrics, ep_sys_metrics)
                

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if "individual_reward" in info[agent_id].keys():
                                idv_rews.append(info[agent_id]["individual_reward"])
                        train_infos[agent_id].update({"individual_rewards": np.mean(idv_rews)})
                        train_infos[agent_id].update(
                            {
                                "average_episode_rewards": np.mean(self.buffer[agent_id].rewards)
                                * self.episode_length
                            }
                        )
                else:
                    for agent_id in range(self.num_agents):
                        avg_rew = np.mean(self.buffer[agent_id].rewards) * self.episode_length
                        train_infos[agent_id].update({"average_episode_rewards": avg_rew})
                    avg_all = np.mean([np.mean(self.buffer[a].rewards) for a in range(self.num_agents)])
                    print("  average reward: {:.4f}".format(avg_all * self.episode_length))

                self.log_train(train_infos, total_num_steps)

                # ---- 全局系统指标写入 tensorboard ----
                mu_L = self.envs.envs[0].env.base.mu_L
                mu_E = self.envs.envs[0].env.base.mu_E
                avg_tc = np.mean(ep_sys_time_costs) if ep_sys_time_costs else 0.0
                avg_en = np.mean(ep_sys_energies) if ep_sys_energies else 0.0
                w_cost_raw = mu_L * avg_tc + mu_E * avg_en
                w_cost_norm = (np.mean(ep_norm_weighted_costs)
                               if ep_norm_weighted_costs else 0.0)
                w_cost_norm_base = (np.mean(ep_norm_weighted_cost_bases)
                                    if ep_norm_weighted_cost_bases else 0.0)
                user_rs = np.mean([np.mean(self.buffer[a].rewards) * self.episode_length
                                   for a in range(self.n_users)])
                uav_rs = np.mean([np.mean(self.buffer[a].rewards) * self.episode_length
                                  for a in range(self.n_users, self.num_agents)])
                self.writter.add_scalars("system/weighted_cost",
                                         {"weighted_cost": w_cost_norm}, total_num_steps)
                self.writter.add_scalars("system/weighted_cost_raw",
                                         {"weighted_cost_raw": w_cost_raw}, total_num_steps)
                self.writter.add_scalars("system/weighted_cost_base",
                                         {"weighted_cost_base": w_cost_norm_base}, total_num_steps)
                self.writter.add_scalars("system/cost_improvement",
                                         {"cost_improvement": np.mean([i[0].get('cost_improvement', 0.0) for i in infos])},
                                         total_num_steps)
                self.writter.add_scalars("system/system_reward_raw",
                                         {"system_reward_raw": np.mean([i[0].get('system_reward_raw', 0.0) for i in infos])},
                                         total_num_steps)
                self.writter.add_scalars("system/avg_delay",
                                         {"avg_delay": np.mean(ep_delays)}, total_num_steps)
                self.writter.add_scalars("system/total_energy",
                                         {"total_energy": avg_en}, total_num_steps)
                avg_fly = np.mean(ep_uav_fly_energies) if ep_uav_fly_energies else 0.0
                self.writter.add_scalars("system/uav_fly_energy",
                                         {"uav_fly_energy": avg_fly}, total_num_steps)
                self.writter.add_scalars("system/avg_user_reward",
                                         {"avg_user_reward": user_rs}, total_num_steps)
                self.writter.add_scalars("system/avg_uav_reward",
                                         {"avg_uav_reward": uav_rs}, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    # =================================================================
    # 可视化与指标辅助方法
    # =================================================================
    def _save_trajectory(self, episode, ep_user_pos, ep_uav_pos):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        user_colors = plt.cm.tab10(np.linspace(0, 1, self.n_users))
        for i in range(self.n_users):
            pos = np.array(ep_user_pos[i])
            ax.plot(pos[:, 0], pos[:, 1], '-', color=user_colors[i],
                    alpha=0.4, linewidth=0.8)
            ax.plot(pos[0, 0], pos[0, 1], 'o', color=user_colors[i], markersize=4)
            ax.plot(pos[-1, 0], pos[-1, 1], 's', color=user_colors[i], markersize=4)

        uav_colors = ['red', 'blue', 'green', 'purple', 'orange']
        uav_markers = ['D', '^', 'v', '<', '>']
        for j in range(self.n_uavs):
            pos = np.array(ep_uav_pos[j])
            c = uav_colors[j % len(uav_colors)]
            m = uav_markers[j % len(uav_markers)]
            # [核心修改] 加了 marker='.' 和 markersize=6，让轨迹线每个时隙打一个点！
            ax.plot(pos[:, 0], pos[:, 1], linestyle='-', marker='.', color=c,
                    linewidth=1.5, markersize=6, alpha=0.8, label=f'UAV {j}')
            ax.plot(pos[0, 0], pos[0, 1], m, color=c, markersize=10)
            ax.plot(pos[-1, 0], pos[-1, 1], '*', color=c, markersize=14)

        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Episode {episode} Trajectories (o=start, */s=end)')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.traj_dir, f'episode_{episode}.png')
        fig.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _print_episode_metrics(self, episode, ep_delays, ep_sys_energies,
                               ep_sys_time_costs=None,
                               ep_uav_fly_energies=None,
                               ep_norm_weighted_costs=None):
        avg_delay = np.mean(ep_delays)
        avg_energy = np.mean(ep_sys_energies)

        cost_str = ""
        if ep_sys_time_costs:
            mu_L = self.envs.envs[0].env.base.mu_L
            mu_E = self.envs.envs[0].env.base.mu_E
            avg_time_cost = np.mean(ep_sys_time_costs)
            weighted_cost = mu_L * avg_time_cost + mu_E * avg_energy
            cost_str = f" | W_Cost_Raw: {weighted_cost:.2f}"

        if ep_norm_weighted_costs:
            avg_norm_cost = np.mean(ep_norm_weighted_costs)
            cost_str += f" | W_Cost_Norm: {avg_norm_cost:.3f}"

        fly_str = ""
        if ep_uav_fly_energies:
            avg_fly = np.mean(ep_uav_fly_energies)
            fly_str = f" | UAV_Fly: {avg_fly:.1f}J"

        user_rews = [np.mean(self.buffer[a].rewards) * self.episode_length
                     for a in range(self.n_users)]
        uav_rews = [np.mean(self.buffer[a].rewards) * self.episode_length
                    for a in range(self.n_users, self.num_agents)]

        print(f"\n  [Ep {episode}] "
              f"Delay: {avg_delay:.4f}s | E_total: {avg_energy:.1f}J{fly_str}{cost_str}")
        print(f"           "
              f"User_R: {np.mean(user_rews):.3f} | UAV_R: {np.mean(uav_rews):.3f}")

    def _save_performance_metrics(self, episode, user_rows, uav_rows, sys_rows):
        """把性能指标保存为独立的 CSV"""
        if not user_rows: return
        pd.DataFrame(user_rows).to_csv(os.path.join(self.metrics_dir, f'ep_{episode}_user_metrics.csv'), index=False)
        pd.DataFrame(uav_rows).to_csv(os.path.join(self.metrics_dir, f'ep_{episode}_uav_metrics.csv'), index=False)
        pd.DataFrame(sys_rows).to_csv(os.path.join(self.metrics_dir, f'ep_{episode}_system_metrics.csv'), index=False)
        print(f"  >>> Saved 3 Performance Tables (User, UAV, Sys) to {self.metrics_dir}")

    def _save_reward_table(self, episode, ep_reward_rows):
        """保存高度拆解的奖励详情表 (带中文公式表头)"""
        path = os.path.join(self.reward_dir, f'episode_{episode}_rewards.csv')
        if not ep_reward_rows:
            return

        # 1. 第一行的中文公式说明
        formula_text = (
            "说明：用户奖励 = 0.4 * 系统奖励 - 0.6 * 个体归一化成本 | "
            "无人机奖励 = 0.3 * 系统奖励 + 0.7 * (混合引导 + 密集区覆盖 + 时延贡献 - 能耗惩罚 - 越界/碰撞惩罚)\n"
        )

        # 2. 转为 pandas DataFrame 并控制列的先后顺序
        df = pd.DataFrame(ep_reward_rows)
        cols_order = [
            'Step', 'Agent_Type', 'Agent_ID', 
            'System_Reward', 'System_Cumulative_Reward', 
            'Agent_Step_Total_Reward', 'Agent_Cumulative_Reward', 'System_Cost_Improve', 'System_Reward_Raw',
            'w1_System_Reward', 'w2_Cost_Improvement', 'Norm_Delay_Ratio', 'Norm_Energy_Ratio', 
            'w_Sys_Part_Reward', 'Centroid_Guide', 'Density_Guide', 'Delay_Benefit',
            'Boundary_Penalty', 'Collision_Penalty', 'UAV_Energy_Penalty',
            'Guide_Mode', 'Assoc_Mix',
        ]
        df = df[cols_order]

        # 3. 写入文件 (使用 utf-8-sig 确保 Excel 打开中文不乱码)
        with open(path, 'w', encoding='utf-8-sig') as f:
            f.write(formula_text)         
            df.to_csv(f, index=False)
        print(f"  >>> Saved Reward Details Table to {self.reward_dir}")

    def _collect_performance_metrics(self, step, info_e0, 
                                     ep_sys_metrics, ep_user_metrics, ep_uav_metrics,
                                     sys_cum_time_cost, sys_cum_energy, user_cum_energy):
        """提取并保存3张表需要的性能数据 (系统、用户、无人机)"""
        # 1. 记录系统级数据 (对应 Eq.28 的目标函数)
        sys_step_time = info_e0[0].get('sys_time_cost', 0.0)
        sys_step_eng = info_e0[0]['total_system_energy']
        sys_cum_time_cost += sys_step_time
        sys_cum_energy += sys_step_eng
        
        w_L = self.Base.mu_L
        w_E = self.Base.mu_E
        obj_val = w_L * sys_step_time + w_E * sys_step_eng
        obj_val_norm = info_e0[0].get('norm_weighted_cost', 0.0)
        obj_val_norm_base = info_e0[0].get('norm_weighted_cost_base', 0.0)
        obj_improve = info_e0[0].get('cost_improvement', 0.0)
        reward_raw = info_e0[0].get('system_reward_raw', 0.0)
        
        ep_sys_metrics.append({
            'Step': step,
            'Sys_Time_Cost': sys_step_time,
            'Sys_Energy_J': sys_step_eng,
            'Cum_Sys_Time_Cost': sys_cum_time_cost,
            'Cum_Sys_Energy_J': sys_cum_energy,
            'Objective_Value_Cost': obj_val,
            'Objective_Value_Norm': obj_val_norm,
            'Objective_Value_Norm_Baseline': obj_val_norm_base,
            'Cost_Improvement': obj_improve,
            'System_Reward_Raw': reward_raw,
            'mu_L': w_L,
            'mu_E': w_E
        })

        # 2. 记录 User 级数据
        for i in range(self.n_users):
            eng = info_e0[i]['energy']
            user_cum_energy[i] += eng
            assoc_id = info_e0[i].get('association', 0)
            assoc_str = "Local" if assoc_id == 0 else f"UAV_{assoc_id-1}"
            
            ep_user_metrics.append({
                'Step': step,
                'User_ID': i,
                'Latency_s': info_e0[i]['delay'],
                'Association': assoc_str,
                'Offload_Ratio': info_e0[i].get('offload_ratio', 0.0),
                'Allocated_Freq_Hz': info_e0[i].get('alloc_freq', 0.0), # <=== 完美接住并存入表格！
                'Energy_J': eng,
                'Cum_Energy_J': user_cum_energy[i]
            })

        # 3. 记录 UAV 级数据
        for j in range(self.n_uavs):
            idx = self.n_users + j
            f_eng = info_e0[idx]['fly_energy']
            c_eng = info_e0[idx]['comp_energy']
            ep_uav_metrics.append({
                'Step': step,
                'UAV_ID': j,
                'Pos_X': info_e0[idx]['position'][0],
                'Pos_Y': info_e0[idx]['position'][1],
                'Fly_Energy_J': f_eng,
                'Comp_Energy_J': c_eng,
                'Total_Energy_J': f_eng + c_eng,
                'Cum_Energy_J': info_e0[idx]['cumulative_energy']
            })
            
        # 必须返回这两个标量累计值，以便外部更新
        return sys_cum_time_cost, sys_cum_energy

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)  # shape = [env_num, agent_num * obs_dim]

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                # TODO 这里改造成自己环境需要的形式即可
                # TODO Here, you can change the action_env to the form you need
                action_env = action
                # raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        # actions: keep as list of per-agent arrays to support heterogeneous action dims
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]
                        ]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == "Discrete":
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    eval_action_env = eval_action

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({"eval_average_episode_rewards": eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render("rgb_array")[0][0]
                all_frames.append(image)

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        action_env = action

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )
