import os
import numpy as np
#rl-gym
class Base:
    def __init__(self):
        # =========================
        # 1. 场景设置 (3 UAV, 10 Users)
        # =========================
        self.n_uavs = 3
        self.n_users = 10
        self.field_X = [0, 1000]
        self.field_Y = [0, 1000]
        self.h = 50              # UAV高度 (m)
        self.time_step = 1.0      # delta_t (s)
        self.coverage_radius = 150.0

        # 初始化布局
        self.uav_init_radius = 350.0    # 无人机等边三角形外接圆半径 (距地图中心)
        self.user_cluster_radius = 50.0  # 用户高斯聚类标准差 (地图中心密集区)
        
        # =========================
        # 2. 归一化参数 (RL收敛核心)
        # =========================
        self.norm_pos = 1000.0
        self.norm_data = 3e6              # 对齐 task_size_max
        self.norm_cycle = 500.0           # 对齐 cycles_max
        self.norm_freq = 10e9             
        
        self.norm_energy_uav = 600.0      
        self.norm_energy_user = 0.5       # 对齐实际用户能耗量级 (~0.01-0.3J)

        # =========================
        # 3. 通信与信道参数 (Urban LoS)
        # =========================
        self.f_c = 2e9            
        self.c = 3e8              
        self.alpha_los = 2.8      
        self.alpha_nlos = 3.5     
        self.a = 9.61
        self.b = 0.16
        self.beta0 = 10**(-60/10)
        self.sigma2 = 1e-13
        self.B_total = 20e6

        # =========================
        # 4. 智能体物理属性
        # =========================
        # --- UAV ---
        self.uav_v_max = 15.0
        self.C_uav = 10e9
        self.xi_m = 8.2e-10
        
        # 飞行能耗参数 (Rotary-Wing)
        self.P0 = 79.86
        self.Pi = 88.63
        self.U_tip = 120
        self.v0 = 4.03
        self.d0 = 0.6
        self.rho = 1.225
        self.s = 0.05
        self.A_rotor = 0.5

        # --- User (MU) ---
        self.p_tx_max = 0.2
        self.C_local = 1e9
        self.k_local = 1e-28
        
        # =========================
        # 5. Gauss-Markov 移动模型参数
        # =========================
        self.mobility_slot = 1.0
        self.user_mean_velocity = 0.5
        self.user_mean_direction = 0.1
        self.user_memory_level_velocity = 0.6  
        self.user_memory_level_direction = 0.8 
        self.user_Gauss_variance_velocity = 0.5
        self.user_Gauss_variance_direction = 0.5

        # =========================
        # 6. 任务生成
        # =========================
        self.task_size_min = 0.5e6   # 500 KB
        self.task_size_max = 3e6     # 3 MB
        self.cycles_min = 200        # cycles/bit
        self.cycles_max = 500        # cycles/bit
        self.latency_max = 1.0       # 1 时隙 = 1 秒

        # =========================
        # 7. 成本函数权重 (对应论文 eq.28)
        # =========================
        self.mu_L = 1.0
        self.mu_E = 0.5

        self.omega_H = 1.2
        self.omega_L = 1.0

        # =========================
        # 8. RL 奖励权重
        # =========================
        # 违规惩罚: violation_rate ∈ [0,1]
        # system_reward = cost_improvement_sys - w_penalty * violation_rate
        # cost_improvement_sys ∈ [-1,+1], system_reward ∈ [-1.5, +1]
        self.w_penalty = 0.5
        self.w_overboundary = 2
        self.w_collision = 1.0

        # 用户奖励:
        #   r_user = w1_user * r_sys + w2_user * indiv_improvement - w_vio_user * violation_i
        #   r_sys ∈ [-1.5, +1], indiv_improvement ∈ [-1,+1]
        self.w1_user = 0.5
        self.w2_user = 0.5
        self.w_vio_user = 0.3

        # UAV奖励:
        #   r_uav = w_sys_uav * r_sys + w_ind_uav * individual_uav
        #   系统奖励主导, 个体引导辅助; 系统收敛后UAV奖励自然趋稳
        self.w_sys_uav = 0.6
        self.w_ind_uav = 0.4

        # 奖励归一化: delay/energy 先clip到同一上限再缩放到[0,1]
        self.reward_norm_cap = 3.0

        # UAV轨迹引导策略: "center" | "association" | "hybrid"
        # hybrid: 未关联时用中心引导, 关联后逐步切向关联用户簇
        self.uav_guide_mode = os.getenv("UAV_GUIDE_MODE", "hybrid").lower()
        if self.uav_guide_mode not in {"center", "association", "hybrid"}:
            self.uav_guide_mode = "hybrid"

        self.w_proximity = 1.0      # step-wise接近奖励 (到达后衰减, 不会在稳态震荡)
        self.w_density = 0.2        # 用户密集区覆盖奖励 (降低, 稳态~0.06恒定量)
        self.w_delay_benefit = 0.3  # 服务时延改善 (降低, 稳态~0.03, 不主导)
        self.w_energy_uav = 0.15    # UAV能耗惩罚 (归一化后~0.02, 保持小权重)
        # =========================
        # 9. 安全约束
        # =========================
        self.uav_safe_dist = 20.0
        self.boundary_warn = 50.0
