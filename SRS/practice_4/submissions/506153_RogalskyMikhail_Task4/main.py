import time
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import mujoco_viewer
import numpy as np

paused = False


# def key_callback(keycode):
#     if chr(keycode) == ' ':
#         global paused
#         paused = not paused


m = mujoco.MjModel.from_xml_path('optimus.xml')
d = mujoco.MjData(m)

def set_torque(mj_data, KP, KV, theta):
    d.ctrl[0] = KP * (-mj_data.qpos[0] + theta) + KV * (0 - mj_data.qvel[0])

SIMEND = 50
TIMESTEP = 0.01
STEP_NUM = int(SIMEND / TIMESTEP)
timeseries = np.linspace(0, SIMEND, STEP_NUM)

#T = 2 # [s]
FREQ = 2.11 # [Hz]
AMP = np.deg2rad(19.83) # [rad]
BIAS = np.deg2rad(23.6) # [rad]

theta_des = AMP * np.sin(FREQ * timeseries) + BIAS


qpos_log = []
qvel_log = []
ctrl_log = []
theta_des_log = []

EE_position_x = []
EE_position_z = []

viewer = mujoco_viewer.MujocoViewer(m,
                                    d,
                                    title="OPTIMUS",
                                    width=1920,
                                    height=1080)

start_time = time.time()

for i in range(STEP_NUM):
    if not viewer.is_alive:
        break

    step_start = time.time()  # время начала шага

    if not paused:
        set_torque(d, 25, 10, theta_des[i])

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


# with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
#     start = time.time()
#     while viewer.is_running():
#         step_start = time.time()
#
#         if not paused:
#             mujoco.mj_step(m, d)
#
#             with viewer.lock():
#                 viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
#
#             viewer.sync()
#             time_until_next_step = m.opt.timestep - (time.time() - step_start)
#             if time_until_next_step > 0:
#                 time.sleep(time_until_next_step)

t = np.arange(len(theta_des_log)) * TIMESTEP  # правильная ось времени

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