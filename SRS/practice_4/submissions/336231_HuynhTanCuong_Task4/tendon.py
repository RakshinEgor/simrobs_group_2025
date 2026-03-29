# %%
import mujoco
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np
import os
from lxml import etree
import mujoco.viewer
import time

# %%
f1 = "/home/huynh/repos/simrobs_group_2025/SRS/practice_4/submissions/336231_HuynhTanCuong_Task4/tendon.xml"
# %%
model = mujoco.MjModel.from_xml_path(f1)
data = mujoco.MjData(model)

def set_torque(mj_data, actuator:int, time, a, f, p):
    mj_data.ctrl[actuator] = a * np.sin(time * f + p)

SIMEND = 20
TIMESTEP = 0.001
STEP_NUM = int(SIMEND / TIMESTEP)
timeseries = np.linspace(0, SIMEND, STEP_NUM)

AMP_1 = 3.616
FREQ_1 = 36.16
BIAS_1 = 3.21

AMP_2 = 4.97
FREQ_2 = 2.24
BIAS_2 = -3.58

sensor_pos_x = []
sensor_pos_z = []

# %%
viewer = mujoco_viewer.MujocoViewer(model, 
                                    data, 
                                    title="4bar", 
                                    width=1920, 
                                    height=1080)

for i in range(STEP_NUM):  
    if viewer.is_alive:     
        set_torque(data, 0, data.time, AMP_1, FREQ_1, BIAS_1)
        set_torque(data, 1, data.time, AMP_2, FREQ_2, BIAS_2)
        
        sensor_pos = data.sensordata[:3]
        sensor_pos_x.append(sensor_pos[0])
        sensor_pos_z.append(sensor_pos[2])
        
        mujoco.mj_step(model, data)
        viewer.render()

    else:
        break
viewer.close()

# %%
midlength = int(STEP_NUM/2)

plt.clf()
plt.plot(sensor_pos_x[50:], sensor_pos_z[50:], '-', linewidth=2, label='P')
plt.title('End-effector trajectory', fontsize=12, fontweight='bold')
plt.legend(loc='upper left')
plt.xlabel('X-Axis [m]')
plt.ylabel('Z-Axis [m]')
plt.axis('equal')
plt.grid()
plt.draw()
