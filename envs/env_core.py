import numpy as np
from envs.Base import Base
from envs.physics_engine import PhysicsEngine


class EnvCore(object):
    """
    UAV辅助医疗IoT任务卸载多智能体环境
    智能体: n_users个用户 + n_uavs个无人机
    用户动作: [卸载比例, 关联服务器logits(local + M个UAV)]
    无人机动作: [速度, 方向, 对每个用户的算力分配]
    """

    def __init__(self):
        self.base = Base()
        self.physics = PhysicsEngine(self.base)

        self.n_users = self.base.n_users
        self.n_uavs = self.base.n_uavs
        self.agent_num = self.n_users + self.n_uavs

        # 统一观测维度 (所有智能体相同, 便于框架处理)
        # all_user_pos(2*U) + all_uav_pos(2*M) + own_energy(1)
        # + all_task_sizes(U) + all_task_cycles(U) + agent_type(1) + agent_id(1)
        # + dist_to_user_centroid(1) + direction_to_centroid(2)
        # + dist_to_nearest_uav(1) + time_progress(1)
        self.obs_dim = (2 * self.n_users + 2 * self.n_uavs
                        + 1 + self.n_users + self.n_users + 1 + 1
                        + 1 + 2
                        + 1 + 1)

        # 异构动作维度
        # 用户: ratio(1) + assoc_logits(1+M) = 1 + 1 + n_uavs
        self.user_action_dim = 1 + 1 + self.n_uavs
        # 无人机: speed(1) + direction(1) + freq_per_user(U) = 2 + n_users
        self.uav_action_dim = 2 + self.n_users
        self.action_dims = ([self.user_action_dim] * self.n_users
                            + [self.uav_action_dim] * self.n_uavs)

        self.max_steps = 60
        self.current_step = 0

        # 记录上一步到目标的距离, 用于step-wise引导奖励
        self._prev_dist_to_target = None

        self.users = None
        self.uavs = None
        self.tasks = None

    # =========================================================
    # 初始化
    # =========================================================
    def reset(self):
        self.current_step = 0
        self._prev_dist_to_target = None
        self._init_users()
        self._init_uavs()
        self._generate_tasks()
        return self._get_obs()

    def _init_users(self):
        self.users = []
        cx = (self.base.field_X[0] + self.base.field_X[1]) / 2
        cy = (self.base.field_Y[0] + self.base.field_Y[1]) / 2
        spread = self.base.user_cluster_radius
        for _ in range(self.n_users):
            x = np.clip(np.random.normal(cx, spread),
                        self.base.field_X[0], self.base.field_X[1])
            y = np.clip(np.random.normal(cy, spread),
                        self.base.field_Y[0], self.base.field_Y[1])
            self.users.append({
                'position': np.array([x, y]),
                'velocity': np.random.uniform(0.0, self.base.user_mean_velocity * 2),
                'direction': np.random.uniform(0, 2 * np.pi),
                'energy': 0.0,
                'priority': np.random.choice([0, 1], p=[0.8, 0.2]),
            })

    def _init_uavs(self):
        self.uavs = []
        cx = (self.base.field_X[0] + self.base.field_X[1]) / 2
        cy = (self.base.field_Y[0] + self.base.field_Y[1]) / 2
        r = self.base.uav_init_radius
        for j in range(self.n_uavs):
            angle = 2 * np.pi * j / self.n_uavs + np.pi / 2
            ux = np.clip(cx + r * np.cos(angle),
                         self.base.field_X[0], self.base.field_X[1])
            uy = np.clip(cy + r * np.sin(angle),
                         self.base.field_Y[0], self.base.field_Y[1])
            self.uavs.append({
                'position': np.array([ux, uy]),
                'energy': 0.0,
                'cumulative_energy': 0.0,
            })

    def _generate_tasks(self):
        self.tasks = []
        for _ in range(self.n_users):
            self.tasks.append({
                'data_size': np.random.uniform(
                    self.base.task_size_min, self.base.task_size_max),
                'cpu_cycles': np.random.uniform(
                    self.base.cycles_min, self.base.cycles_max),
                'deadline': self.base.latency_max,
            })

    # =========================================================
    # 观测构建
    # =========================================================
    def _get_obs(self):
        user_pos = np.array([u['position'] for u in self.users]).flatten() / self.base.norm_pos
        uav_pos_raw = np.array([u['position'] for u in self.uavs])
        uav_pos = uav_pos_raw.flatten() / self.base.norm_pos
        task_sizes = np.array([t['data_size'] for t in self.tasks]) / self.base.norm_data
        task_cycles = np.array([t['cpu_cycles'] for t in self.tasks]) / self.base.norm_cycle

        # 计算用户中心 (归一化)
        user_pos_raw = np.array([u['position'] for u in self.users])
        user_centroid = np.mean(user_pos_raw, axis=0)

        # 时间进度 [0, 1]
        time_progress = self.current_step / max(self.max_steps, 1)

        obs_list = []
        for i in range(self.n_users):
            pos = self.users[i]['position']
            vec_to_center = user_centroid - pos
            dist_to_center = np.linalg.norm(vec_to_center)
            if dist_to_center > 1e-3:
                dir_to_center = vec_to_center / dist_to_center
            else:
                dir_to_center = np.array([0.0, 0.0])

            # 到最近UAV的距离 (关键特征: 帮助用户决定是否卸载)
            dists_to_uavs = np.linalg.norm(uav_pos_raw - pos, axis=1)
            min_dist_to_uav = np.min(dists_to_uavs)

            obs = np.concatenate([
                user_pos,
                uav_pos,
                [self.users[i]['energy'] / self.base.norm_energy_user],
                task_sizes,
                task_cycles,
                [0.0],
                [i / self.agent_num],
                [dist_to_center / self.base.norm_pos],
                dir_to_center,
                [min_dist_to_uav / self.base.norm_pos],
                [time_progress],
            ]).astype(np.float32)
            obs_list.append(obs)

        for j in range(self.n_uavs):
            pos = self.uavs[j]['position']
            vec_to_center = user_centroid - pos
            dist_to_center = np.linalg.norm(vec_to_center)
            if dist_to_center > 1e-3:
                dir_to_center = vec_to_center / dist_to_center
            else:
                dir_to_center = np.array([0.0, 0.0])

            # UAV: 到最近用户的距离
            dists_to_users = np.linalg.norm(user_pos_raw - pos, axis=1)
            min_dist_to_user = np.min(dists_to_users)

            obs = np.concatenate([
                user_pos,
                uav_pos,
                [self.uavs[j]['energy'] / self.base.norm_energy_uav],
                task_sizes,
                task_cycles,
                [1.0],
                [(self.n_users + j) / self.agent_num],
                [dist_to_center / self.base.norm_pos],
                dir_to_center,
                [min_dist_to_user / self.base.norm_pos],
                [time_progress],
            ]).astype(np.float32)
            obs_list.append(obs)

        return obs_list

    # =========================================================
    # 环境步进
    # =========================================================
    def step(self, actions):
        # ---------- 1. 解析动作 ----------
        user_ratios, user_assocs = self._parse_user_actions(actions)
        uav_speeds, uav_dirs, uav_freqs = self._parse_uav_actions(actions)

        # ---------- 2. 更新位置 ----------
        self._update_uav_positions(uav_speeds, uav_dirs)
        self.physics.update_user_positions(self.users)

        # ---------- 3. 统计每个UAV关联的用户 ----------
        users_per_uav = [[] for _ in range(self.n_uavs)]
        for i in range(self.n_users):
            if user_assocs[i] > 0:
                users_per_uav[user_assocs[i] - 1].append(i)

        # ---------- 4. 计算时延和能耗 ----------
        (user_delays, user_energies, uav_comp_energies,
         uav_fly_energies, deadline_violations) = self._compute_delays_and_energies(
            user_ratios, user_assocs, uav_freqs, uav_speeds, users_per_uav)

        # ---------- 5. 计算奖励 ----------
        rewards, reward_details, system_metrics = self._compute_rewards(
            user_delays, user_energies, deadline_violations,
            uav_fly_energies, uav_comp_energies, users_per_uav)

        # ---------- 6. 新任务 + 新观测 ----------
        self._generate_tasks()
        self.current_step += 1
        obs_list = self._get_obs()

        done = self.current_step >= self.max_steps
        dones = [done] * self.agent_num

        # [修改这里] 传入 user_ratios 和 user_assocs
        infos = self._build_infos(
            user_delays, user_energies, deadline_violations,
            uav_fly_energies, uav_comp_energies, reward_details,
            user_ratios, user_assocs, uav_freqs, system_metrics)

        return [obs_list, rewards, dones, infos]

    # =========================================================
    # 动作解析
    # =========================================================
    def _parse_user_actions(self, actions):
        user_ratios = []
        user_assocs = []
        for i in range(self.n_users):
            act = actions[i]
            ratio = _sigmoid(act[0])
            assoc = int(np.argmax(act[1:1 + 1 + self.n_uavs]))
            if assoc == 0:
                ratio = 0.0
            user_ratios.append(ratio)
            user_assocs.append(assoc)
        return user_ratios, user_assocs

    def _parse_uav_actions(self, actions):
        uav_speeds = []
        uav_dirs = []
        uav_freqs = []
        for j in range(self.n_uavs):
            act = actions[self.n_users + j]
            speed = _sigmoid(act[0]) * self.base.uav_v_max
            direction = np.tanh(act[1]) * np.pi
            raw_f = act[2:2 + self.n_users]
            exp_f = np.exp(raw_f - np.max(raw_f))
            freq = (exp_f / np.sum(exp_f)) * self.base.C_uav
            uav_speeds.append(speed)
            uav_dirs.append(direction)
            uav_freqs.append(freq)
        return uav_speeds, uav_dirs, uav_freqs

    # =========================================================
    # 位置更新
    # =========================================================
    def _update_uav_positions(self, uav_speeds, uav_dirs):
        for j in range(self.n_uavs):
            dx = uav_speeds[j] * np.cos(uav_dirs[j]) * self.base.time_step
            dy = uav_speeds[j] * np.sin(uav_dirs[j]) * self.base.time_step
            new_pos = self.uavs[j]['position'] + np.array([dx, dy])
            new_pos[0] = np.clip(new_pos[0], self.base.field_X[0], self.base.field_X[1])
            new_pos[1] = np.clip(new_pos[1], self.base.field_Y[0], self.base.field_Y[1])
            self.uavs[j]['position'] = new_pos

    # =========================================================
    # 时延 & 能耗计算 (论文公式 14-24)
    # =========================================================
    def _compute_delays_and_energies(self, user_ratios, user_assocs,
                                     uav_freqs, uav_speeds, users_per_uav):
        user_delays = np.zeros(self.n_users)
        user_energies = np.zeros(self.n_users)
        uav_comp_energies = np.zeros(self.n_uavs)
        uav_fly_energies = np.zeros(self.n_uavs)
        deadline_violations = np.zeros(self.n_users)

        for i in range(self.n_users):
            D = self.tasks[i]['data_size']
            C = self.tasks[i]['cpu_cycles']
            tau = self.tasks[i]['deadline']
            lam = user_ratios[i]
            assoc = user_assocs[i]
            local_frac = 1.0 - lam

            # --- 本地计算 (eq.14, eq.19) ---
            T_local = 0.0
            E_local = 0.0
            if local_frac > 1e-8:
                T_local = local_frac * D * C / self.base.C_local
                E_local = self.base.k_local * (self.base.C_local ** 2) * local_frac * D * C

            # --- 卸载计算 (eq.15, eq.20) ---
            T_off = 0.0
            E_tx = 0.0
            if assoc > 0 and lam > 1e-8:
                uav_idx = assoc - 1
                g = self.physics.get_channel_gain(
                    self.users[i]['position'], self.uavs[uav_idx]['position'])
                n_assoc = max(len(users_per_uav[uav_idx]), 1)
                bw = self.base.B_total / n_assoc
                R = self.physics.compute_rate(g, bw, self.base.p_tx_max)

                if R > 1e-3:
                    T_tx = lam * D / R
                else:
                    T_tx = tau * 100.0

                f_mu = max(uav_freqs[uav_idx][i], 1e3)
                T_comp_uav = lam * D * C / f_mu
                T_off = T_tx + T_comp_uav

                E_tx = self.base.p_tx_max * T_tx
                uav_comp_energies[uav_idx] += self.base.xi_m * lam * D * C

            # --- 任务完成时延 (eq.17) ---
            T_u = max(T_local, T_off)
            T_u = min(T_u, tau * 5.0)
            user_delays[i] = T_u

            # --- 用户总能耗 (eq.21) ---
            E_u = E_local + E_tx
            user_energies[i] = E_u
            self.users[i]['energy'] = E_u

            if T_u > tau:
                deadline_violations[i] = 1.0

        # --- 无人机飞行能耗 (eq.22, eq.24) ---
        for j in range(self.n_uavs):
            E_fly = self.physics.compute_uav_energy(uav_speeds[j])
            uav_fly_energies[j] = E_fly
            E_total = E_fly + uav_comp_energies[j]
            self.uavs[j]['energy'] = E_total
            self.uavs[j]['cumulative_energy'] += E_total

        return (user_delays, user_energies, uav_comp_energies,
                uav_fly_energies, deadline_violations)

    # =========================================================
    # 奖励计算  (论文目标: min 总成本 = μ_L·时延 + μ_E·能耗)
    # =========================================================
    def _compute_rewards(self, user_delays, user_energies, violations,
                         uav_fly_e, uav_comp_e, users_per_uav):
        """
        重新设计的奖励函数 — 解决卸载永远不如本地的核心问题:
        1. 系统奖励: 只比较用户侧成本(时延+用户能耗), 不含UAV能耗
           (UAV能耗由UAV个体奖励中轻量惩罚即可, 不应污染系统基线)
        2. 用户奖励: 纯时延改善信号(卸载的核心优势), 能耗作为小惩罚
        3. UAV奖励: 引导飞行为主, 覆盖+服务质量为辅
        """
        b = self.base
        eps = 1e-6
        cap = max(float(getattr(b, 'reward_norm_cap', 3.0)), 1.0)
        progress = self.current_step / max(self.max_steps, 1)

        # =========================================================
        # 1. 计算每个用户的本地基线和实际时延/能耗
        # =========================================================
        baseline_delays = np.zeros(self.n_users)   # 本地全部计算的时延
        baseline_energies = np.zeros(self.n_users)  # 本地全部计算的能耗

        for i in range(self.n_users):
            D = self.tasks[i]['data_size']
            C = self.tasks[i]['cpu_cycles']
            baseline_delays[i] = D * C / b.C_local
            baseline_energies[i] = b.k_local * (b.C_local ** 2) * D * C

        # 优先级权重
        omegas = np.array([b.omega_H if self.users[i]['priority'] == 1 else b.omega_L
                           for i in range(self.n_users)])

        # =========================================================
        # 2. 系统奖励: 只比较用户侧的加权时延 (卸载的核心优势维度)
        #    不含UAV能耗, 避免基线不公平
        # =========================================================
        # 加权时延: omega_i * T_i / T_max
        weighted_actual_delays = omegas * np.minimum(user_delays / b.latency_max, cap)
        weighted_baseline_delays = omegas * np.minimum(baseline_delays / b.latency_max, cap)

        avg_actual_delay = float(np.mean(weighted_actual_delays))
        avg_baseline_delay = float(np.mean(weighted_baseline_delays))

        # 时延改善率: 正=卸载更快, 负=卸载更慢
        delay_improvement = np.clip(
            (avg_baseline_delay - avg_actual_delay) / max(avg_baseline_delay, eps),
            -1.0, 1.0)

        violation_rate = float(np.mean(violations))

        # 系统奖励 = 时延改善 - 违约惩罚
        # 时延改善放大1.5x, 违约惩罚随episode进度增大
        # 整体scale控制在[-1, +1.5]左右, 避免critic学习困难
        vio_weight = b.w_penalty * (0.3 + 0.7 * progress)
        system_reward = float(delay_improvement * 1.5 - vio_weight * violation_rate)

        # =========================================================
        # 2b. 用于日志的完整成本指标 (包含UAV能耗, 仅供监控)
        # =========================================================
        uav_energy_norms = self._norm_to_unit((uav_fly_e + uav_comp_e) / b.norm_energy_uav, cap)
        hover_energy_norm = float(self._norm_to_unit(
            self.physics.compute_uav_energy(0.0) / b.norm_energy_uav, cap))

        user_e_norms = np.array([self._norm_to_unit(user_energies[i] / b.norm_energy_user, cap)
                                 for i in range(self.n_users)])
        base_e_norms = np.array([self._norm_to_unit(baseline_energies[i] / b.norm_energy_user, cap)
                                 for i in range(self.n_users)])

        avg_user_e_cost = float(np.mean(user_e_norms))
        avg_uav_e_cost = float(np.mean(uav_energy_norms))
        avg_baseline_user_e = float(np.mean(base_e_norms))

        norm_weighted_cost = (b.mu_L * avg_actual_delay
                              + b.mu_E * (avg_user_e_cost + avg_uav_e_cost))
        norm_weighted_cost_base = (b.mu_L * avg_baseline_delay
                                   + b.mu_E * (avg_baseline_user_e + hover_energy_norm))

        energy_improvement = np.clip(
            (avg_baseline_user_e - avg_user_e_cost) / max(avg_baseline_user_e, eps),
            -1.0, 1.0)
        cost_improvement_sys = np.clip(
            (norm_weighted_cost_base - norm_weighted_cost) / max(norm_weighted_cost_base, eps),
            -1.0, 1.0)

        rewards = []
        reward_details = []

        # =========================================================
        # 3. 用户奖励: 时延改善 + 轻量能耗惩罚 + 违约惩罚
        # =========================================================
        user_positions = np.array([u['position'] for u in self.users])
        uav_positions = np.array([u['position'] for u in self.uavs])

        # 预计算: 每个用户到最近UAV的距离
        min_dist_to_uav = np.zeros(self.n_users)
        for i in range(self.n_users):
            dists = np.linalg.norm(uav_positions - user_positions[i], axis=1)
            min_dist_to_uav[i] = np.min(dists)

        for i in range(self.n_users):
            D = self.tasks[i]['data_size']
            C = self.tasks[i]['cpu_cycles']
            omega = omegas[i]

            # 个体时延改善率 (卸载核心信号)
            delay_ratio = np.clip(
                (baseline_delays[i] - user_delays[i]) / max(baseline_delays[i], eps),
                -1.0, 1.0)

            # 用户能耗改善 (轻量信号, 卸载减少本地计算->减少能耗)
            energy_ratio = np.clip(
                (baseline_energies[i] - user_energies[i]) / max(baseline_energies[i], eps),
                -1.0, 1.0)

            violation_i = float(violations[i])

            # 奖励 = 系统奖励 + 个体时延改善 + 轻量能耗改善 - 违约惩罚
            w1_sys = 0.3 * system_reward
            w2_delay = 0.5 * delay_ratio * omega  # 高优先级用户时延改善加权更大
            w3_energy = 0.1 * energy_ratio

            # 违约惩罚: UAV远时减轻 (因为是UAV没到位导致的)
            vio_pen = b.w_vio_user * violation_i
            if violation_i > 0 and min_dist_to_uav[i] > b.coverage_radius * 2.0:
                vio_pen *= 0.2

            r = np.clip(w1_sys + w2_delay + w3_energy - vio_pen, -2.0, 2.0)
            rewards.append([r])
            reward_details.append({
                'agent_type': 'user',
                'system_reward': float(system_reward),
                'system_reward_raw': float(system_reward),
                'delay_saving': float(delay_improvement),
                'energy_improvement': float(energy_improvement),
                'energy_penalty': float(avg_user_e_cost + avg_uav_e_cost),
                'violation_rate': violation_rate,
                'w1_system': float(w1_sys),
                'cost_improvement': float(delay_ratio),
                'cost_penalty': 0.0,
                'delay_ratio': float(self._norm_to_unit(user_delays[i] / b.latency_max, cap)),
                'energy_ratio': float(self._norm_to_unit(user_energies[i] / b.norm_energy_user, cap)),
                'w2_improvement': float(w2_delay),
                'norm_weighted_cost': float(norm_weighted_cost),
                'total': float(r),
            })

        # =========================================================
        # 4. UAV 奖励: 引导飞行 + 覆盖 + 服务质量
        # =========================================================
        user_centroid = np.mean(user_positions, axis=0)
        half_diag = np.sqrt((b.field_X[1] - b.field_X[0]) ** 2
                            + (b.field_Y[1] - b.field_Y[0]) ** 2) / 2.0

        current_dists = [0.0] * self.n_uavs

        for j in range(self.n_uavs):
            pos = self.uavs[j]['position']
            target_pos, assoc_mix = self._get_uav_guidance_target(
                j, users_per_uav, user_positions, user_centroid)

            dist_to_target = np.linalg.norm(pos - target_pos)

            # === Step-wise 接近奖励 ===
            if self._prev_dist_to_target is not None and j < len(self._prev_dist_to_target):
                approach_delta = self._prev_dist_to_target[j] - dist_to_target
            else:
                approach_delta = 0.0
            approach_reward = np.clip(approach_delta / b.uav_v_max, -1.0, 1.0)
            dist_decay = min(dist_to_target / b.coverage_radius, 1.0)
            step_guide = approach_reward * dist_decay

            # === 静态距离惩罚 ===
            dist_penalty = -min(dist_to_target / b.uav_init_radius, 1.0)

            guide_reward = b.w_proximity * (0.5 * step_guide + 0.5 * dist_penalty)

            # === 覆盖用户数奖励 ===
            dists_to_users = np.linalg.norm(user_positions - pos, axis=1)
            n_covered = int(np.sum(dists_to_users < b.coverage_radius))
            coverage_reward = 0.4 * (n_covered / max(self.n_users / self.n_uavs, 1.0))
            coverage_reward = min(coverage_reward, 0.6)

            # === 密度引导 ===
            avg_dist_to_users = float(np.mean(dists_to_users))
            guide_sigma = 200.0
            density_proximity = np.exp(-0.5 * (avg_dist_to_users / guide_sigma) ** 2)
            density_reward = b.w_density * density_proximity

            current_dists[j] = dist_to_target

            # === 服务收益 ===
            delay_benefit = 0.0
            if users_per_uav[j]:
                delay_reds = []
                for uid in users_per_uav[j]:
                    d_to_uav = np.linalg.norm(self.users[uid]['position'] - pos)
                    if d_to_uav > b.coverage_radius * 1.5:
                        continue
                    delay_red = np.clip(
                        (baseline_delays[uid] - user_delays[uid]) / max(baseline_delays[uid], eps),
                        -1.0, 1.0)
                    delay_reds.append(omegas[uid] * delay_red)
                if delay_reds:
                    delay_benefit = float(np.mean(delay_reds))

            uav_total_e = uav_fly_e[j] + uav_comp_e[j]
            energy_pen = b.w_energy_uav * self._norm_to_unit(uav_total_e / b.norm_energy_uav, cap)

            boundary_pen = 0.0
            dist_to_edge = min(
                pos[0] - b.field_X[0], b.field_X[1] - pos[0],
                pos[1] - b.field_Y[0], b.field_Y[1] - pos[1])
            if dist_to_edge <= b.boundary_warn:
                boundary_pen = b.w_overboundary * (b.boundary_warn - dist_to_edge) / b.boundary_warn

            collision_terms = []
            for k in range(self.n_uavs):
                if k == j:
                    continue
                d = np.linalg.norm(pos - self.uavs[k]['position'])
                if d < b.uav_safe_dist:
                    collision_terms.append((b.uav_safe_dist - d) / b.uav_safe_dist)
            collision_pen = b.w_collision * (float(np.mean(collision_terms)) if collision_terms else 0.0)

            delay_reward = b.w_delay_benefit * delay_benefit
            uav_individual = (guide_reward + density_reward + coverage_reward
                              + delay_reward
                              - energy_pen - boundary_pen - collision_pen)

            # UAV奖励: 早期以引导飞行为主(w_ind=0.8), 后期增加系统奖励(w_ind=0.4)
            ind_weight = 0.8 - 0.4 * progress
            sys_weight = 1.0 - ind_weight
            w_sys_part = sys_weight * system_reward
            w_ind_part = ind_weight * uav_individual
            r_uav = np.clip(w_sys_part + w_ind_part, -2.0, 2.0)
            rewards.append([r_uav])
            reward_details.append({
                'agent_type': 'uav',
                'system_reward': float(system_reward),
                'system_reward_raw': float(system_reward),
                'delay_saving': float(delay_improvement),
                'energy_improvement': float(energy_improvement),
                'energy_penalty_sys': float(avg_user_e_cost + avg_uav_e_cost),
                'w_sys_part': float(w_sys_part),
                'centroid_guide': float(guide_reward),
                'density_guide': float(density_reward),
                'delay_benefit': float(delay_reward),
                'energy_pen': float(energy_pen),
                'boundary_pen': float(boundary_pen),
                'collision_pen': float(collision_pen),
                'uav_individual': float(uav_individual),
                'w_ind_part': float(w_ind_part),
                'guide_mode': str(getattr(b, 'uav_guide_mode', 'hybrid')),
                'assoc_mix': float(assoc_mix),
                'target_dist_norm': float(dist_to_target / max(half_diag, eps)),
                'norm_weighted_cost': float(norm_weighted_cost),
                'coverage_count': n_covered,
                'total': float(r_uav),
            })

        self._prev_dist_to_target = current_dists

        system_metrics = {
            'delay_improvement': float(delay_improvement),
            'energy_improvement': float(energy_improvement),
            'norm_delay_cost': float(avg_actual_delay),
            'norm_energy_cost': float(avg_user_e_cost + avg_uav_e_cost),
            'norm_weighted_cost': float(norm_weighted_cost),
            'norm_weighted_cost_base': float(norm_weighted_cost_base),
            'violation_rate': float(violation_rate),
            'cost_improvement': float(cost_improvement_sys),
            'system_reward_raw': float(system_reward),
            'system_reward': float(system_reward),
        }
        return rewards, reward_details, system_metrics

    def _norm_to_unit(self, value, cap):
        arr = np.asarray(value, dtype=np.float64)
        clipped = np.clip(arr, 0.0, cap) / cap
        if np.isscalar(value):
            return float(clipped)
        return clipped

    def _get_uav_guidance_target(self, uav_id, users_per_uav, user_positions, global_centroid):
        b = self.base
        assoc_users = users_per_uav[uav_id]
        if assoc_users:
            assoc_pos = user_positions[assoc_users]
            assoc_weights = np.array(
                [b.omega_H if self.users[uid]['priority'] == 1 else b.omega_L
                 for uid in assoc_users],
                dtype=np.float64)
            assoc_centroid = np.average(assoc_pos, axis=0, weights=assoc_weights)
        else:
            assoc_centroid = global_centroid

        mode = str(getattr(b, 'uav_guide_mode', 'hybrid')).lower()
        if mode == 'center':
            return global_centroid, 0.0
        if mode == 'association':
            return assoc_centroid, 1.0 if assoc_users else 0.0

        # hybrid: 关联用户越多, 目标越偏向关联簇; 无关联时自动退化为中心引导
        ref_assoc = max(1.0, self.n_users / max(self.n_uavs, 1))
        assoc_mix = np.clip(len(assoc_users) / ref_assoc, 0.0, 1.0)
        target = (1.0 - assoc_mix) * global_centroid + assoc_mix * assoc_centroid
        return target, float(assoc_mix)

    # =========================================================
    # 辅助
    # =========================================================
    def _build_infos(self, user_delays, user_energies, violations,
                     uav_fly_e, uav_comp_e, reward_details, user_ratios,
                     user_assocs, uav_freqs, system_metrics):
        total_sys_energy = float(np.sum(user_energies) + np.sum(uav_fly_e + uav_comp_e))
        avg_delay = float(np.mean(user_delays))
        
        # [核心新增] 计算系统总时间成本 (考虑高低优先级的加权时延)
        sys_time_cost = 0.0
        for i in range(self.n_users):
            omega = self.base.omega_H if self.users[i]['priority'] == 1 else self.base.omega_L
            sys_time_cost += omega * user_delays[i]

        infos = []
        for i in range(self.n_users):
            # [核心新增] 提取无人机给该用户分配的算力
            assoc = user_assocs[i]
            if assoc > 0:
                uav_idx = assoc - 1
                alloc_f = max(uav_freqs[uav_idx][i], 1e3) # 对应你底层的计算
            else:
                alloc_f = 0.0 # 如果是在本地计算，无人机分配的频率就是0
                
            infos.append({
                'delay': float(user_delays[i]),
                'energy': float(user_energies[i]),
                'violation': float(violations[i]),
                'position': self.users[i]['position'].copy(),
                'reward_details': reward_details[i],
                'total_system_energy': total_sys_energy,
                'sys_time_cost': float(sys_time_cost), # 新增系统时间成本
                'avg_user_delay': avg_delay,
                'offload_ratio': float(user_ratios[i]), # 新增卸载比
                'association': int(user_assocs[i]),     # 新增关联对象
                'alloc_freq': float(alloc_f), # <=== 新增：将分配的频率打包传出！
                'norm_delay_cost': float(system_metrics['norm_delay_cost']),
                'norm_energy_cost': float(system_metrics['norm_energy_cost']),
                'norm_weighted_cost': float(system_metrics['norm_weighted_cost']),
                'norm_weighted_cost_base': float(system_metrics['norm_weighted_cost_base']),
                'violation_rate': float(system_metrics['violation_rate']),
                'cost_improvement': float(system_metrics['cost_improvement']),
                'system_reward_raw': float(system_metrics['system_reward_raw']),
            })
        for j in range(self.n_uavs):
            infos.append({
                'fly_energy': float(uav_fly_e[j]),
                'comp_energy': float(uav_comp_e[j]),
                'cumulative_energy': float(self.uavs[j]['cumulative_energy']),
                'position': self.uavs[j]['position'].copy(),
                'reward_details': reward_details[self.n_users + j],
                'total_system_energy': total_sys_energy,
                'sys_time_cost': float(sys_time_cost),
                'avg_user_delay': avg_delay,
                'norm_delay_cost': float(system_metrics['norm_delay_cost']),
                'norm_energy_cost': float(system_metrics['norm_energy_cost']),
                'norm_weighted_cost': float(system_metrics['norm_weighted_cost']),
                'norm_weighted_cost_base': float(system_metrics['norm_weighted_cost_base']),
                'violation_rate': float(system_metrics['violation_rate']),
                'cost_improvement': float(system_metrics['cost_improvement']),
                'system_reward_raw': float(system_metrics['system_reward_raw']),
            })
        return infos


def _sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))
