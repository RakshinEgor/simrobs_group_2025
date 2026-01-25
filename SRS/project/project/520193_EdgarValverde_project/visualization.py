import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import core_optimization as core

def set_axes_equal(ax):
    """Sets 3D axes to equal scale so the robot doesn't look distorted."""
    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()

    ranges = [abs(x_lim[1]-x_lim[0]), abs(y_lim[1]-y_lim[0]), abs(z_lim[1]-z_lim[0])]
    radius = 0.5 * max(ranges)
    
    mid_x = np.mean(x_lim)
    mid_y = np.mean(y_lim)
    mid_z = np.mean(z_lim)

    ax.set_xlim3d([mid_x - radius, mid_x + radius])
    ax.set_ylim3d([mid_y - radius, mid_y + radius])
    ax.set_zlim3d([mid_z - radius, mid_z + radius])

def plot_results(u_opt, history, args, limits, method_name="Optimization"):
    (bot, q_start, _, _, A, B, deep, dgrs, n_steps, _) = args
    
    # 1. Reconstruct Full Trajectory State
    traj = core.sim_trajectory(u_opt, q_start, A, B, deep, dgrs, n_steps)
    
    dt = 0.05 # Must match main.py (or be passed in args)
    time_vec = np.arange(len(traj)) * dt
    
    u_reshaped = u_opt.reshape(n_steps, dgrs)
    u_plot = np.vstack([u_reshaped, u_reshaped[-1, :]])

    # --- FIGURE 1: KINEMATIC INTEGRATION PROFILES ---
    fig1, axs = plt.subplots(5, 6, figsize=(18, 10))
    # AGREGADO: method_name en el título
    fig1.suptitle(f"{method_name} - ABB IRB 140: Kinematic Profiles", fontsize=16)
    
    ylabels = ["Control (Snap)\n[rad/s^4]", "Jerk\n[rad/s^3]", "Accel\n[rad/s^2]", "Velocity\n[rad/s]", "Position\n[rad]"]
    
    for j_idx in range(dgrs): 
        idx = j_idx * deep
        pos = traj[:, idx]
        vel = traj[:, idx+1]
        acc = traj[:, idx+2]
        jerk = traj[:, idx+3]
        snap = u_plot[:, j_idx]
        
        data_rows = [snap, jerk, acc, vel, pos]
        
        for row in range(5):
            ax = axs[row, j_idx]
            
            if row == 0:
                ax.step(time_vec, data_rows[row], where='post', linewidth=1.5, color='#d62728')
            else:
                ax.plot(time_vec, data_rows[row], linewidth=1.5, color='#1f77b4')
            
            ax.grid(True, linestyle=':', alpha=0.5)
            
            if j_idx == 0: 
                ax.set_ylabel(ylabels[row], fontsize=9, fontweight='bold')
            if row == 0: 
                ax.set_title(f"Joint {j_idx + 1}", fontsize=11, fontweight='bold')
            if row == 4: 
                ax.set_xlabel("Time (s)", fontsize=9)
            else:
                ax.set_xticklabels([]) 

            if row == 4: # Position Limits
                ax.axhline(limits['q_max'][j_idx], color='r', linestyle='--', alpha=0.6)
                ax.axhline(limits['q_min'][j_idx], color='r', linestyle='--', alpha=0.6)
            
            if row == 3: # Velocity Limits
                ax.axhline(limits['v_max'][j_idx], color='orange', linestyle='--', alpha=0.8)
                ax.axhline(-limits['v_max'][j_idx], color='orange', linestyle='--', alpha=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- FIGURE 2: 3D TRAJECTORY ---
    fig2 = plt.figure(figsize=(10, 8))
    ax3d = fig2.add_subplot(111, projection='3d')
    
    coords_start = bot.get_stick_model(q_start)
    ax3d.plot(coords_start[:,0], coords_start[:,1], coords_start[:,2], 
              'o--', color='gray', alpha=0.5, label='Start')
    
    q_final = [traj[-1, i*deep] for i in range(dgrs)]
    coords_end = bot.get_stick_model(q_final)
    ax3d.plot(coords_end[:,0], coords_end[:,1], coords_end[:,2], 
              'o-', color='black', linewidth=3, markersize=5, label='Final Pose')
    
    tcp_path = []
    for k in range(len(traj)):
        q_k = [traj[k, i*deep] for i in range(dgrs)]
        T = bot.forward_kinematics(q_k)
        tcp_path.append(T[:3, 3])
        
        if k % 5 == 0:
            R = T[:3, :3]
            p = T[:3, 3]
            scale = 0.05
            ax3d.quiver(p[0],p[1],p[2], R[0,0],R[1,0],R[2,0], color='r', length=scale)
            ax3d.quiver(p[0],p[1],p[2], R[0,1],R[1,1],R[2,1], color='g', length=scale)
            ax3d.quiver(p[0],p[1],p[2], R[0,2],R[1,2],R[2,2], color='b', length=scale)
            
    tcp_path = np.array(tcp_path)
    ax3d.plot(tcp_path[:,0], tcp_path[:,1], tcp_path[:,2], 
              color='magenta', linewidth=2, label='TCP Path')
    
    ax3d.set_title(f"{method_name} - 3D Cartesian Trajectory")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    set_axes_equal(ax3d)
    plt.legend()
    
    plt.show()

def plot_convergence_log(history, method_name="Optimization"):
    if not history:
        return

    iterations = [h['iter'] for h in history]
    pos_errors = [h['err_pos'] for h in history]
    rot_errors = [h['err_rot'] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{method_name} - Convergence Analysis (Log Scale)", fontsize=16)

    ax1.semilogy(iterations, pos_errors, 'b.-', linewidth=1.5, markersize=8)
    ax1.set_ylabel("Position Error [meters] (Log)", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", ls="-", alpha=0.4)
    ax1.set_title("Cartesian Position Error vs Iterations", fontsize=12)
    
    final_pos = pos_errors[-1]
    ax1.annotate(f"{final_pos:.2e} m", xy=(iterations[-1], final_pos), 
                 xytext=(iterations[-1], final_pos*1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    ax2.semilogy(iterations, rot_errors, 'r.-', linewidth=1.5, markersize=8)
    ax2.set_ylabel("Orientation Error [norm] (Log)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Iterations / Generations", fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.4)
    ax2.set_title("Orientation Error vs Iterations", fontsize=12)

    final_rot = rot_errors[-1]
    ax2.annotate(f"{final_rot:.2e}", xy=(iterations[-1], final_rot), 
                 xytext=(iterations[-1], final_rot*1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def export_csv(u_opt, args, filename="trajectory_data.csv"):
    (bot, q_start, _, _, A, B, deep, dgrs, n_steps, _) = args
    traj = core.sim_trajectory(u_opt, q_start, A, B, deep, dgrs, n_steps)
    data = {}
    for i in range(dgrs):
        data[f'q{i+1}'] = traj[:, i*deep]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ CSV Exported successfully: {filename}")