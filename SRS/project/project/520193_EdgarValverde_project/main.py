import numpy as np
import time  # <--- IMPORTANTE: Importar librería time
import robot
import state_space
import solvers
import visualization

# --- ROBOT CONFIGURATION (ABB IRB 140) ---
# Parameters extracted from PDF Table III and Dimensions (Page 2)
# Units: Meters and Radians.
# L0=70, L1=352, L2=360, L3=254, L4=126, L5=65 (mm) -> converted to meters
dh_irb140 = {
    # Joint 1: d=L1 (0.352), a=L0 (0.070)
    1: {'theta': np.pi/2, 'd': 0.352, 'a': 0.070, 'alpha': np.pi/2},
    # Joint 2: a=L2 (0.360)
    2: {'theta': np.pi/2, 'd': 0.0,   'a': 0.360, 'alpha': 0.0},
    # Joint 3: 
    3: {'theta': 0.0,     'd': 0.0,   'a': 0.0,   'alpha': np.pi/2},
    # Joint 4: d=L3+L4 (0.254+0.126 = 0.380)
    4: {'theta': 0.0,     'd': 0.380, 'a': 0.0,   'alpha': -np.pi/2},
    # Joint 5:
    5: {'theta': 0.0,     'd': 0.0,   'a': 0.0,   'alpha': np.pi/2},
    # Joint 6: d=L5 (0.065) - assuming tool offset is separate
    6: {'theta': 0.0,     'd': 0.065, 'a': 0.0,   'alpha': 0.0}
}

bot = robot.Robot(dh_irb140)

# --- 2. STATE SPACE CONFIGURATION ---
dt = 0.2        # 50ms time step (finer resolution)
deep = 4        # Control Input: Snap (4th derivative) -> [Pos, Vel, Acc, Jerk]
dgrs = 6        # 6 DOF
n_steps = 10    # Total time = 2.0 seconds (assuming dt*n_steps)

# Generate Matrices
A, B = state_space.get_matrices(deep, dgrs, dt)

# --- 3. TRAJECTORY GOALS ---
# Initial State (Home)
q_start = np.zeros(dgrs) 

# Target Cartesian Pose [x, y, z, roll, pitch, yaw]
# Goal: Move to front-right and rotate wrist
target_pos = [-0.45, 0.15, 0.40] 
target_ori = [0, np.radians(60), 0] 
p_goal = np.array(target_pos + target_ori)

# Final Velocity (Zero to stop)
v_goal = np.zeros(dgrs) 

# Optimization Weights
weights = {
    'w_pos': 80.0,   # Accuracy Priority
    'w_rot': 40.0,   # Orientation Priority
    'w_vel': 10.0,   # Stop condition Priority
    'w_effort': 0.05 # Smoothness (Minimize Snap)
}

args = (bot, q_start, p_goal, v_goal, A, B, deep, dgrs, n_steps, weights)

# --- CONSTRAINTS ---
# Limits in Radians and Rad/s
limits = {
    'q_min': np.radians([-180, -100, -230, -200, -115, -400]),
    'q_max': np.radians([ 180,  110,   50,  200,  115,  400]),
    'v_max': np.radians([200, 200, 260, 360, 360, 450])
}

# --- SOLVER EXECUTION ---
print(f"--- Starting Optimization for ABB IRB 140 ---")
print(f"Target: {target_pos} m")

# Strategy: Use SLSQP directly (Fastest for single trajectory)
# If it fails, enable GA by setting solver_type = 'GA'
solver_type = 'GA' 

u0 = np.zeros(dgrs * n_steps) # Initial guess: Zero Snap
history = []

start_time = time.time()

if solver_type == 'SLSQP':
    u_opt = solvers.solve_slsqp(u0, args, limits, history)
else:
    u_bounds = (-50, 50) # Snap bounds
    u_opt, history = solvers.solve_ga(args, limits, u_bounds, n_gen=200)

end_time = time.time()
execution_time = end_time - start_time

print(f"\n⏱️  Solver Execution Time: {execution_time:.4f} seconds")
print(f"--------------------------------------------------")

# --- VISUALIZATION & EXPORT ---
print("Generating Plots...")
visualization.plot_results(u_opt, history, args, limits, method_name=solver_type)
visualization.plot_convergence_log(history, method_name=solver_type)
visualization.export_csv(u_opt, args)