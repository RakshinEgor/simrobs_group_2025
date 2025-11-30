import mujoco
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

model = mujoco.MjModel.from_xml_path('laba_4.xml')
data = mujoco.MjData(model)

def set_torque_R1(mj_data, KP, KV, desired_pos):
    current_pos = mj_data.qpos[0]  
    current_vel = mj_data.qvel[0]  
    data.ctrl[0] = KP * (desired_pos - current_pos) + KV * (0 - current_vel)

def set_torque_R2(mj_data, KP, KV, desired_pos):
    current_pos = mj_data.qpos[1]  
    current_vel = mj_data.qvel[1]  
    data.ctrl[1] = KP * (desired_pos - current_pos) + KV * (0 - current_vel)


SIMEND = 30  
TIMESTEP = 0.001
STEP_NUM = int(SIMEND / TIMESTEP)
timeseries = np.linspace(0, SIMEND, STEP_NUM)

AMP1 = np.deg2rad(34.45)
FREQ1 = 3.85
BIAS1 = np.deg2rad(42.1)
theta1 = AMP1 * np.sin(2 * np.pi * FREQ1 * timeseries) + BIAS1

AMP2 = np.deg2rad(11.77)
FREQ2 = 1.61
BIAS2 = np.deg2rad(-31.6)
theta2 = AMP2 * np.sin(2 * np.pi * FREQ2 * timeseries) + BIAS2

position_time = []
R1_position_x = []
R1_position_z = []
R2_position_x = []
R2_position_z = []
theta1_trajectory = []
theta2_trajectory = []


viewer = mujoco_viewer.MujocoViewer(model, data, title="Laba_4", width=1920, height=1080)


for i in range(STEP_NUM):
    if viewer.is_alive:
        
        set_torque_R1(data, 100, 10, theta1[i])
        set_torque_R2(data, 100, 10, theta2[i])
        
      
        current_time = data.time
        position_time.append(current_time)
        
        
        position_R1 = data.site_xpos[model.site('R1_site').id]
        R1_position_x.append(position_R1[0])
        R1_position_z.append(position_R1[2])
        
       
        position_R2 = data.site_xpos[model.site('R2_site').id]
        R2_position_x.append(position_R2[0])
        R2_position_z.append(position_R2[2])
        
        theta1_trajectory.append(theta1[i])
        theta2_trajectory.append(theta2[i])
       
        
       
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

viewer.close()
