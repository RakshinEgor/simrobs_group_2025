import numpy as np
from robot import euler_to_rot_matrix

def sim_trajectory(u_flat, q_start, A, B, deep, dgrs, n_steps):
    """
    Simulates the system evolution.
    Returns array of shape (n_steps+1, total_states).
    """
    u = u_flat.reshape(n_steps, dgrs)
    
    # Initial state x0: [q1, q1_dot, q1_ddot, q1_dddot, q2...]
    x = np.zeros(dgrs * deep)
    for i in range(dgrs):
        x[i*deep] = q_start[i] # Set initial position, assume vel/acc=0
        
    trajectory = [x.copy()]
    curr_x = x.copy()
    
    for k in range(n_steps):
        curr_x = A @ curr_x + B @ u[k, :]
        trajectory.append(curr_x.copy())
        
    return np.array(trajectory)

def get_detailed_errors(u_flat, args):
    (bot, q_start, target_pose, target_vel, A, B, deep, dgrs, n_steps, _) = args
    
    traj = sim_trajectory(u_flat, q_start, A, B, deep, dgrs, n_steps)
    final_state = traj[-1]
    
    q_final = np.zeros(dgrs)
    dq_final = np.zeros(dgrs)
    for i in range(dgrs):
        q_final[i] = final_state[i*deep]
        dq_final[i] = final_state[i*deep+1]
        
    T_final = bot.forward_kinematics(q_final)
    p_final = T_final[:3, 3]
    R_final = T_final[:3, :3]
    
    p_des = target_pose[:3]
    err_pos = np.linalg.norm(p_des - p_final)
    
    # || I - R_des * R_final.T ||
    r_des, p_des_ang, y_des = target_pose[3], target_pose[4], target_pose[5]
    R_des = euler_to_rot_matrix(r_des, p_des_ang, y_des)
    err_rot = np.linalg.norm(np.eye(3) - R_des @ R_final.T)
    
    err_vel = np.linalg.norm(dq_final - target_vel)

    return err_pos, err_rot, err_vel

def objective_function(u_flat, args):
    (bot, q_start, target_pose, target_vel, A, B, deep, dgrs, n_steps, weights) = args
    
    traj = sim_trajectory(u_flat, q_start, A, B, deep, dgrs, n_steps)
    final_state = traj[-1]
    
    q_final = np.zeros(dgrs)
    dq_final = np.zeros(dgrs)
    
    for i in range(dgrs):
        q_final[i] = final_state[i*deep]       # Position
        dq_final[i] = final_state[i*deep+1]    # Velocity
        
    T_final = bot.forward_kinematics(q_final)
    p_final = T_final[:3, 3]
    R_final = T_final[:3, :3]
    
    p_des = target_pose[:3]
    err_pos = np.linalg.norm(p_des - p_final)
    
    r_des, p_des_ang, y_des = target_pose[3], target_pose[4], target_pose[5]
    R_des = euler_to_rot_matrix(r_des, p_des_ang, y_des)
    err_rot = np.linalg.norm(np.eye(3) - R_des @ R_final.T)
    
    err_vel = np.linalg.norm(dq_final - target_vel)
    
    effort = np.sum(u_flat**2)
    
    loss = (weights['w_pos'] * err_pos + 
            weights['w_rot'] * err_rot + 
            weights['w_vel'] * err_vel + 
            weights['w_effort'] * effort)
            
    return loss

def get_constraints_violation(u_flat, args, limits):
    """Calculates sum of violations for Penalty method."""
    (_, q_start, _, _, A, B, deep, dgrs, n_steps, _) = args
    traj = sim_trajectory(u_flat, q_start, A, B, deep, dgrs, n_steps)
    
    violation = 0.0
    
    for i in range(dgrs):
        idx_pos = i * deep
        idx_vel = i * deep + 1
        
        q_hist = traj[:, idx_pos]
        dq_hist = traj[:, idx_vel]
        
        violation += np.sum(np.maximum(0, limits['q_min'][i] - q_hist))
        violation += np.sum(np.maximum(0, q_hist - limits['q_max'][i]))
        violation += np.sum(np.maximum(0, np.abs(dq_hist) - limits['v_max'][i]))
        
    return violation