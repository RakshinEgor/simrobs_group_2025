import time
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import mujoco_viewer
import numpy as np

paused = False

m = mujoco.MjModel.from_xml_path('optimus.xml')
d = mujoco.MjData(m)

def set_torque(mj_data, KP, KV, theta):
    d.ctrl[0] = KP * (-mj_data.qpos[0] + theta) + KV * (0 - mj_data.qvel[0])

SIMEND = 50
TIMESTEP = 0.01
STEP_NUM = int(SIMEND / TIMESTEP)
timeseries = np.linspace(0, SIMEND, STEP_NUM)

# القيم المعدلة فقط - بدون أي تغييرات أخرى
FREQ = 1.48  # [Hz] - تم التعديل
AMP = np.deg2rad(57.89)  # [rad] - تم التعديل
BIAS = np.deg2rad(7.7)   # [rad] - تم التعديل

theta_des = AMP * np.sin(FREQ * timeseries) + BIAS

qpos_log = []
qvel_log = []
ctrl_log = []
theta_des_log = []

EE_position_x = []
EE_position_z = []

viewer = mujoco_viewer.MujocoViewer(m, d, title="OPTIMUS",  width=1920, height=1080)

start_time = time.time()

for i in range(STEP_NUM):
    if not viewer.is_alive:
        break

    step_start = time.time()  # وقت بداية الخطوة

    if not paused:
        set_torque(d, 25, 10, theta_des[i])  # نفس معاملات التحكم

        mujoco.mj_step(m, d)
        viewer.render()

        qpos_log.append(d.qpos[0])
        qvel_log.append(d.qvel[0])
        ctrl_log.append(d.ctrl[0])
        position_EE = d.site_xpos[1]
        EE_position_x.append(position_EE[0])
        EE_position_z.append(position_EE[2])
        theta_des_log.append(theta_des[i])

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

viewer.close()

t = np.arange(len(theta_des_log)) * TIMESTEP

midlength = int(STEP_NUM/2)

plt.plot(EE_position_x[600:], EE_position_z[600:], '-', linewidth=2, label='P')
plt.title('End-effector trajectory', fontsize=12, fontweight='bold')
plt.legend(loc='upper left')
plt.xlabel('X-Axis [m]')
plt.ylabel('Z-Axis [m]')
plt.axis('equal')
plt.grid()
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(t, theta_des_log, label="θ_desired", linewidth=2)
plt.plot(t, qpos_log, label="θ_actual (qpos)", alpha=0.8)
plt.title("Отработка PD-регулятора")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.grid(True)
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(t, qvel_log, label="qvel", color="green")
plt.title("Угловая скорость")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [rad/s]")
plt.grid(True)
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(t, ctrl_log, label="Torque ctrl", color="red")
plt.title("Управляющий момент")
plt.xlabel("Time [s]")
plt.ylabel("Torque")
plt.grid(True)
plt.legend()

plt.show()