import mujoco
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np
import time
from tendon import generate_tendon_xml

# -----------------------------
# 1. Parameters & XML Generation
# -----------------------------
R1, R2 = 0.013, 0.045
a, b, c = 0.040, 0.042, 0.087

xml = generate_tendon_xml(R1, R2, a, b, c)
model = mujoco.MjModel.from_xml_string(xml.encode("utf-8"))
data = mujoco.MjData(model)

# -----------------------------
# 2. Joint Control Parameters (Your Specifications)
# -----------------------------
# Joint 1 parameters
AMP1 = np.radians(17.9)    # Convert degrees to radians
FREQ1 = 3.82               # Hz
BIAS1 = np.radians(-36.5)  # Convert degrees to radians

# Joint 2 parameters  
AMP2 = np.radians(43.56)   # Convert degrees to radians
FREQ2 = 3.81               # Hz
BIAS2 = np.radians(-14.7)  # Convert degrees to radians

# -----------------------------
# 3. REVISED Control Function - Direct Kinematics Approach
# -----------------------------
def calculate_desired_angles(time):
    """Calculate desired joint angles based on specifications"""
    q1_desired = BIAS1 + AMP1 * np.sin(2 * np.pi * FREQ1 * time)
    q2_desired = BIAS2 + AMP2 * np.sin(2 * np.pi * FREQ2 * time)
    return q1_desired, q2_desired

def calculate_tendon_lengths(q1, q2):
    """
    Calculate tendon lengths based on joint angles and pulley geometry
    This is a simplified model - you may need to adjust based on your actual setup
    """
    # For a 2-joint tendon system, the relationship is:
    # L1 = L10 - R1*q1 - R2*q2  (for one tendon)
    # L2 = L20 + R1*q1 - R2*q2  (for the other tendon)
    
    L10 = 0.2  # Nominal length of tendon 1
    L20 = 0.2  # Nominal length of tendon 2
    
    L1 = L10 - R1 * q1 - R2 * q2
    L2 = L20 + R1 * q1 - R2 * q2
    
    return L1, L2

def set_tendon_control(mj_data, time):
    """
    Direct tendon control based on desired kinematics
    """
    # Get desired joint angles
    q1_des, q2_des = calculate_desired_angles(time)
    
    # Calculate desired tendon lengths
    L1_des, L2_des = calculate_tendon_lengths(q1_des, q2_des)
    
    # Get current tendon lengths from simulation
    L1_curr = mj_data.ten_length[0]
    L2_curr = mj_data.ten_length[1]
    
    # PD control on tendon lengths
    kp_tendon = 1000  # Proportional gain for tendon length control
    kd_tendon = 50    # Derivative gain
    
    # Tendon length errors
    error_L1 = L1_des - L1_curr
    error_L2 = L2_des - L2_curr
    
    # Tendon velocity (approximate)
    vel_L1 = -mj_data.ten_velocity[0] if len(mj_data.ten_velocity) > 0 else 0
    vel_L2 = -mj_data.ten_velocity[1] if len(mj_data.ten_velocity) > 1 else 0
    
    # PD control
    data.ctrl[0] = kp_tendon * error_L1 + kd_tendon * (0 - vel_L1)
    data.ctrl[1] = kp_tendon * error_L2 + kd_tendon * (0 - vel_L2)
    
    # Apply control limits
    data.ctrl[0] = np.clip(data.ctrl[0], -100, 100)
    data.ctrl[1] = np.clip(data.ctrl[1], -100, 100)

# -----------------------------
# 4. Alternative: Simple Antagonistic Control
# -----------------------------
def set_simple_control(mj_data, time):
    """
    Simple antagonistic control with your specifications
    """
    # Calculate desired joint motions
    q1_motion = AMP1 * np.sin(2 * np.pi * FREQ1 * time)
    q2_motion = AMP2 * np.sin(2 * np.pi * FREQ2 * time)
    
    # Combined control signal
    # Tendon 1 pulls for positive motion, Tendon 2 pulls for negative motion
    control_gain = 50
    
    # Map joint motions to tendon forces
    data.ctrl[0] = control_gain * (q1_motion + 0.5 * q2_motion)
    data.ctrl[1] = control_gain * (-q1_motion + 0.5 * q2_motion)
    
    # Apply limits
    data.ctrl[0] = np.clip(data.ctrl[0], -80, 80)
    data.ctrl[1] = np.clip(data.ctrl[1], -80, 80)

# -----------------------------
# 5. Simulation Parameters
# -----------------------------
SIMEND = 5.0  # 1 second to see several cycles
TIMESTEP = 0.001
STEP_NUM = int(SIMEND / TIMESTEP)

# Arrays to store data
sensor_pos_x = []
sensor_pos_z = []
joint1_angles = []
joint2_angles = []
desired_joint1 = []
desired_joint2 = []
time_history = []
tendon_forces = [[], []]
tendon_lengths = [[], []]

# -----------------------------
# 6. Initialize Simulation
# -----------------------------
print("Initializing simulation...")
# Reset to zero position first
data.qpos[0] = 0  # Joint 1
data.qpos[1] = 0  # Joint 2

# Let it settle
for _ in range(500):
    mujoco.mj_step(model, data)

print("Starting main simulation...")

# -----------------------------
# 7. Run Simulation
# -----------------------------
try:
    viewer = mujoco_viewer.MujocoViewer(model, data, title="2R Tendon Robot - Revised Control", width=1200, height=800)
    
    for i in range(STEP_NUM):
        if viewer.is_alive:
            current_time = data.time
            
            # Use simple control method (more reliable)
            set_simple_control(data, current_time)
            
            # Store data
            sensor_pos = data.sensordata[:3]
            sensor_pos_x.append(sensor_pos[0])
            sensor_pos_z.append(sensor_pos[2])
            
            # Store joint angles
            joint1_angles.append(data.qpos[0])
            joint2_angles.append(data.qpos[1])
            
            # Calculate and store desired joint angles
            q1_des, q2_des = calculate_desired_angles(current_time)
            desired_joint1.append(q1_des)
            desired_joint2.append(q2_des)
            
            # Store tendon data
            tendon_forces[0].append(data.ctrl[0])
            tendon_forces[1].append(data.ctrl[1])
            if len(data.ten_length) >= 2:
                tendon_lengths[0].append(data.ten_length[0])
                tendon_lengths[1].append(data.ten_length[1])
            
            time_history.append(current_time)
            
            # Simulation step
            mujoco.mj_step(model, data)
            viewer.render()
            
        else:
            break

    viewer.close()
    print("Simulation completed successfully!")

except Exception as e:
    print(f"Error during simulation: {e}")
    if 'viewer' in locals():
        viewer.close()

# -----------------------------
# 8. Analysis and Plotting
# -----------------------------
if len(time_history) == 0:
    print("No data collected!")
    exit()

# Convert to degrees for plotting
joint1_angles_deg = np.degrees(joint1_angles)
joint2_angles_deg = np.degrees(joint2_angles)
desired_joint1_deg = np.degrees(desired_joint1)
desired_joint2_deg = np.degrees(desired_joint2)

print(f"Simulation collected {len(time_history)} data points")
print(f"Final time: {time_history[-1]:.3f}s")

# Calculate statistics
error1 = np.array(desired_joint1_deg) - np.array(joint1_angles_deg)
error2 = np.array(desired_joint2_deg) - np.array(joint2_angles_deg)
rms_error1 = np.sqrt(np.mean(error1**2))
rms_error2 = np.sqrt(np.mean(error2**2))

print(f"Joint 1 RMS Error: {rms_error1:.2f}°")
print(f"Joint 2 RMS Error: {rms_error2:.2f}°")
print(f"Joint 1 Range: {np.min(joint1_angles_deg):.1f}° to {np.max(joint1_angles_deg):.1f}°")
print(f"Joint 2 Range: {np.min(joint2_angles_deg):.1f}° to {np.max(joint2_angles_deg):.1f}°")

# -----------------------------
# 9. Create Clean Plots
# -----------------------------
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: End-effector trajectory
axes[0,0].plot(sensor_pos_x, sensor_pos_z, 'b-', linewidth=1.5, alpha=0.7)
axes[0,0].set_xlabel('X Position (m)')
axes[0,0].set_ylabel('Z Position (m)')
axes[0,0].set_title('End-effector Trajectory')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].axis('equal')

# Plot 2: Joint 1 tracking
axes[0,1].plot(time_history, desired_joint1_deg, 'r--', linewidth=1, label='Desired')
axes[0,1].plot(time_history, joint1_angles_deg, 'b-', linewidth=1.5, label='Actual', alpha=0.8)
axes[0,1].set_xlabel('Time (s)')
axes[0,1].set_ylabel('Joint Angle (deg)')
axes[0,1].set_title(f'Joint 1 Tracking (RMS Error: {rms_error1:.1f}°)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Joint 2 tracking
axes[1,0].plot(time_history, desired_joint2_deg, 'r--', linewidth=1, label='Desired')
axes[1,0].plot(time_history, joint2_angles_deg, 'g-', linewidth=1.5, label='Actual', alpha=0.8)
axes[1,0].set_xlabel('Time (s)')
axes[1,0].set_ylabel('Joint Angle (deg)')
axes[1,0].set_title(f'Joint 2 Tracking (RMS Error: {rms_error2:.1f}°)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Control forces
axes[1,1].plot(time_history, tendon_forces[0], 'r-', linewidth=1, label='Tendon 1', alpha=0.7)
axes[1,1].plot(time_history, tendon_forces[1], 'b-', linewidth=1, label='Tendon 2', alpha=0.7)
axes[1,1].set_xlabel('Time (s)')
axes[1,1].set_ylabel('Control Force (N)')
axes[1,1].set_title('Tendon Control Forces')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Specifications Used ===")
print(f"Joint 1: {AMP1/np.pi*180:.1f}° amplitude, {FREQ1} Hz, {BIAS1/np.pi*180:.1f}° bias")
print(f"Joint 2: {AMP2/np.pi*180:.1f}° amplitude, {FREQ2} Hz, {BIAS2/np.pi*180:.1f}° bias")