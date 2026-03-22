"""Deep debug: check physical feasibility of offloading"""
import numpy as np
from envs.Base import Base
from envs.physics_engine import PhysicsEngine

b = Base()
pe = PhysicsEngine(b)

print("=== Physical Feasibility Analysis ===")
print(f"Field: {b.field_X[1]}x{b.field_Y[1]}m, UAV height: {b.h}m")
print(f"Coverage radius: {b.coverage_radius}m, UAV init radius: {b.uav_init_radius}m")
print(f"UAV max speed: {b.uav_v_max}m/s, episode length: 60 steps")
print(f"Max UAV travel in 60 steps: {b.uav_v_max * 60}m")
print(f"Steps to reach center from init: {b.uav_init_radius / b.uav_v_max:.1f} steps")

print(f"\nTask size: [{b.task_size_min/1e6:.1f}, {b.task_size_max/1e6:.1f}] MB")
print(f"CPU cycles: [{b.cycles_min}, {b.cycles_max}] cycles/bit")
print(f"Deadline: {b.latency_max}s")

# Check local compute feasibility
print("\n=== Local Compute Feasibility ===")
for D_mb in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    D = D_mb * 1e6
    for C in [200, 350, 500]:
        T_local = D * C / b.C_local
        E_local = b.k_local * (b.C_local ** 2) * D * C
        viol = "VIOLATE" if T_local > b.latency_max else "OK"
        print(f"  D={D_mb:.1f}MB, C={C}: T_local={T_local:.3f}s, E_local={E_local:.6f}J [{viol}]")

# Check offload feasibility at various distances
print("\n=== Offload Feasibility (1 user per UAV, full BW) ===")
user_pos = np.array([500.0, 500.0])  # center
for dist in [50, 100, 150, 200, 300, 400]:
    uav_pos = np.array([500.0 + dist, 500.0])
    g = pe.get_channel_gain(user_pos, uav_pos)
    R = pe.compute_rate(g, b.B_total, b.p_tx_max)  # full bandwidth
    print(f"\n  Distance={dist}m:")
    print(f"    Channel gain={g:.2e}, Rate={R/1e6:.2f} Mbps")
    for D_mb in [0.5, 1.5, 3.0]:
        D = D_mb * 1e6
        for C in [200, 500]:
            T_tx = D / R if R > 1e-3 else 999
            f_uav = b.C_uav  # full compute to 1 user
            T_comp = D * C / f_uav
            T_off = T_tx + T_comp
            T_local = D * C / b.C_local
            speedup = T_local / T_off if T_off > 0 else 0
            viol = "VIOLATE" if T_off > b.latency_max else "OK"
            print(f"    D={D_mb}MB,C={C}: T_tx={T_tx:.4f}s, T_comp={T_comp:.4f}s, T_off={T_off:.4f}s (local={T_local:.3f}s, speedup={speedup:.1f}x) [{viol}]")

# Check with bandwidth sharing (multiple users per UAV)
print("\n=== Offload with BW sharing (dist=100m, C=350) ===")
for n_users_per_uav in [1, 2, 3, 5, 10]:
    bw = b.B_total / n_users_per_uav
    uav_pos = np.array([600.0, 500.0])
    g = pe.get_channel_gain(user_pos, uav_pos)
    R = pe.compute_rate(g, bw, b.p_tx_max)
    f_per_user = b.C_uav / n_users_per_uav  # equal freq
    D = 1.5e6
    C = 350
    T_tx = D / R if R > 1e-3 else 999
    T_comp = D * C / f_per_user
    T_off = T_tx + T_comp
    T_local = D * C / b.C_local
    viol = "VIOLATE" if T_off > b.latency_max else "OK"
    print(f"  {n_users_per_uav} users: BW={bw/1e6:.1f}MHz, Rate={R/1e6:.2f}Mbps, f={f_per_user/1e9:.1f}GHz, T_off={T_off:.4f}s [{viol}]")

# Check UAV energy
print("\n=== UAV Energy ===")
for v in [0, 5, 10, 15]:
    E = pe.compute_uav_energy(v)
    print(f"  v={v}m/s: E_fly={E:.2f}J (normalized: {E/b.norm_energy_uav:.4f})")

# Check norm_weighted_cost baseline range
print("\n=== Baseline Cost Distribution (1000 samples) ===")
costs = []
for _ in range(1000):
    D = np.random.uniform(b.task_size_min, b.task_size_max)
    C = np.random.uniform(b.cycles_min, b.cycles_max)
    T_base = D * C / b.C_local
    E_base = b.k_local * (b.C_local ** 2) * D * C
    d_norm = min(T_base / b.latency_max, 3.0) / 3.0
    e_norm = min(E_base / b.norm_energy_user, 3.0) / 3.0
    cost = b.mu_L * d_norm + b.mu_E * e_norm
    costs.append(cost)

costs = np.array(costs)
print(f"  Baseline cost: mean={costs.mean():.4f}, std={costs.std():.4f}, min={costs.min():.4f}, max={costs.max():.4f}")
print(f"  Fraction violating deadline: {np.mean([1 for c in costs if c > 0]):.3f}")

# Check how many tasks violate locally
violations = 0
for _ in range(10000):
    D = np.random.uniform(b.task_size_min, b.task_size_max)
    C = np.random.uniform(b.cycles_min, b.cycles_max)
    T_local = D * C / b.C_local
    if T_local > b.latency_max:
        violations += 1
print(f"\n  Local compute violation rate (10000 samples): {violations/10000:.3f}")
