import mujoco
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import mujoco.viewer
import time

f1 = "MujocoLab4.xml"

model = mujoco.MjModel.from_xml_path(f1)
data = mujoco.MjData(model)

def set_torque1(mj_data, KP, KV, theta):
    data.ctrl[0] = KP * (-mj_data.qpos[0] + theta) + KV * (0 - mj_data.qvel[0])

def set_torque2(mj_data, KP, KV, theta):
    data.ctrl[1] = KP * (-mj_data.qpos[0] + theta) + KV * (0 - mj_data.qvel[0])

SIMEND = 50
TIMESTEP = 0.001
STEP_NUM = int(SIMEND / TIMESTEP)
timeseries = np.linspace(0, SIMEND, STEP_NUM)

FREQ = 3.99 # [Hz]
AMP = 37.89 # [rad]
BIAS = 24 # [rad]

theta_des1 = AMP * np.sin(FREQ * timeseries) + BIAS

FREQ = 3.19 # [Hz]
AMP = 52.89 # [rad]
BIAS = 40.4 # [rad]

theta_des2 = AMP * np.sin(FREQ * timeseries) + BIAS

position_time = []

R1_position_x = []
R1_position_z = []

R2_position_x = []
R2_position_z = []

theta_des1_trajectory = []
theta_des2_trajectory = []

viewer = mujoco_viewer.MujocoViewer(model, 
                                    data, 
                                    title="tendons", 
                                    width=1920, 
                                    height=1080)

for i in range(STEP_NUM):  
    if viewer.is_alive:
        set_torque1(data, 100, 100, theta_des1[i])
        set_torque2(data, 100, 100, theta_des2[i])

        current_time = data.time
        position_time.append(current_time)

        position_R1 = data.site_xpos[5]
        R1_position_x.append(position_R1[0])
        R1_position_z.append(position_R1[2])

        position_R2 = data.site_xpos[12]
        R2_position_x.append(position_R2[0])
        R2_position_z.append(position_R2[2])

        theta_des1_trajectory.append(theta_des1[i])
        theta_des2_trajectory.append(theta_des2[i])

        mujoco.mj_step(model, data)
        viewer.render()

    else:
        break
viewer.close()

midlength = int(STEP_NUM/2)

df = pd.DataFrame({
    'time': position_time,
    'R1_x': R1_position_x,
    'R1_z': R1_position_z,
    'R2_x': R2_position_x,
    'R2_z': R2_position_z,
    'theta_des1': theta_des1,
    'theta_des2': theta_des2
})

csv_filename = "Task4_positions_data.csv"
df.to_csv(csv_filename, index=False)


