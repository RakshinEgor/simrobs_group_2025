import mujoco
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load Model ---
f1 = "lab4_new.xml"
model = mujoco.MjModel.from_xml_path(f1)
data = mujoco.MjData(model)

# --- 2. Simulation & Control Parameters ---
SIMEND = 20.0
TIMESTEP = 0.001
STEP_NUM = int(SIMEND / TIMESTEP)
model.opt.timestep = TIMESTEP 

# PD Controller Gains 
KP = 100.0  # Proportional Gain (Stiffness)
KD = 50.0   # Derivative Gain (Damping)

# --- Trajectory Parameters ---

# I scaled them down by 0.1 for safety.
AMP_1 = 21.6 * 0.1  
FREQ_1 = 2.35
BIAS_1 = 23.2 * 0.1  

AMP_2 = 57.45 * 0.1
FREQ_2 = 2.74
BIAS_2 = -8.1 * 0.1 

# Storage for plotting
sensor_pos_x = []
sensor_pos_z = []
ref_log = [] # To store desired trajectory
act_log = [] # To store actual trajectory

# --- 3. PD Control Function ---
def calculate_pd_output(kp, kd, current_val, current_vel, target_val, target_vel):
    """
    Computes Force = Kp * error + Kd * error_derivative
    """
    error = target_val - current_val
    error_dot = target_vel - current_vel
    u = (kp * error) + (kd * error_dot)
    return u

# --- 4. Main Simulation Loop ---
viewer = mujoco_viewer.MujocoViewer(model, data, title="Lab 4 Simulation", width=1200, height=900)

print("Starting simulation...")

for i in range(STEP_NUM):  
    if viewer.is_alive:
        # A. Get Current Time
        t = data.time
        
        # B. Get Sensor Readings (Feedback)
        # Using name lookups is safer than index [0] or [1]
        q1_curr = data.sensor('t1_pos').data[0]
        dq1_curr = data.sensor('t1_vel').data[0]
        
        q2_curr = data.sensor('t2_pos').data[0]
        dq2_curr = data.sensor('t2_vel').data[0]
        
        # C. Calculate Desired State (Reference Trajectory)
        # Desired Position: A * sin(wt) + bias
        q1_des = AMP_1 * np.sin(FREQ_1 * t) + BIAS_1
        q2_des = AMP_2 * np.sin(FREQ_2 * t) + BIAS_2
        
        # Desired Velocity: derivative of position -> A * w * cos(wt)
        dq1_des = AMP_1 * FREQ_1 * np.cos(FREQ_1 * t)
        dq2_des = AMP_2 * FREQ_2 * np.cos(FREQ_2 * t)
        
        # D. Calculate PD Control Effort
        ctrl1 = calculate_pd_output(KP, KD, q1_curr, dq1_curr, q1_des, dq1_des)
        ctrl2 = calculate_pd_output(KP, KD, q2_curr, dq2_curr, q2_des, dq2_des)
        
        # E. Apply Control to Actuators
        data.ctrl[0] = ctrl1
        data.ctrl[1] = ctrl2

        # F. Step Simulation
        mujoco.mj_step(model, data)
        viewer.render()
        
        # G. Log Data for Plotting
        # End effector position (first sensor in XML)
        ee_pos = data.sensor('ee_pos').data
        sensor_pos_x.append(ee_pos[0])
        sensor_pos_z.append(ee_pos[2])
        
        # Log actuator 1 tracking for debugging
        ref_log.append(q1_des)
        act_log.append(q1_curr)

    else:
        break

viewer.close()

# --- 5. Plotting ---
plt.figure(figsize=(10, 8))

# Subplot 1: End Effector X-Z Trajectory
plt.subplot(2, 1, 1)
# Skip first 50 steps to let simulation settle
start_idx = 50 if len(sensor_pos_x) > 50 else 0
plt.plot(sensor_pos_x[start_idx:], sensor_pos_z[start_idx:], '-', linewidth=2, label='End Effector')
plt.title('End-Effector Trajectory (XZ)', fontsize=12, fontweight='bold')
plt.xlabel('X-Axis [m]')
plt.ylabel('Z-Axis [m]')
plt.axis('equal')
plt.grid(True)
plt.legend()

# Subplot 2: PD Tracking (Check if we are following the sine wave)
plt.subplot(2, 1, 2)
plt.plot(ref_log[start_idx:], 'r--', label='Reference (Desired)')
plt.plot(act_log[start_idx:], 'b-', label='Actual (Sensor)')
plt.title('Actuator 1 Tracking Performance', fontsize=12)
plt.xlabel('Steps')
plt.ylabel('Tendon Length')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()