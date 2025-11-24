import time

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np


MODEL_PATH = 'optimus_knee.xml'

model = mujoco.MjModel.from_xml_path(filename=MODEL_PATH, assets={})
data = mujoco.MjData(model)

prev_err = 0
# kp = 20
# kd = 4

kp = 60
kd = 40

q_log = []
q_des_log = []
err_log = []
t = []


def sine_control(m: mujoco.MjModel, d: mujoco.MjData):
    global prev_err, kp, kd, q_log, q_des_log, err_log, t

    q = d.sensor("O_ang").data[0]
    q_des = np.deg2rad(34.76 * np.sin(3.84 * d.time) + 25.4)

    err = q_des - q
    err_d = (err - prev_err) / m.opt.timestep

    u = kp * err + kd * err_d
    u = np.clip(u, -100, 100)

    d.ctrl[0] = u

    prev_err = err

    q_log.append(np.rad2deg(q))
    q_des_log.append(np.rad2deg(q_des))
    err_log.append(np.rad2deg(err))
    t.append(d.time)


def look_at_xz(viewer: mujoco.viewer.Handle):
    viewer.cam.lookat = [0, 0, 0.05]
    viewer.cam.distance = 0.75
    viewer.cam.elevation = 0


with mujoco.viewer.launch_passive(model, data) as viewer:
    look_at_xz(viewer)

    mujoco.set_mjcb_control(sine_control)
    start = time.time()

    while viewer.is_running():
        mujoco.mj_step(model, data)

        viewer.sync()

plt.figure(figsize=(16, 8))
plt.plot(t, q_log, color='blue', linewidth=2, label='O(t) fact')
plt.plot(t, q_des_log, color='red', linewidth=2, linestyle='--', label='O(t) expected')
plt.xlabel('t, s')
plt.ylabel('O(t), deg')
plt.title('Фактический и желаемый угол O(t)')
plt.legend()
plt.grid(True)

plt.figure(figsize=(16, 8))
plt.plot(t, err_log, color='green', linewidth=2, label='e(t)')
plt.xlabel('t, s')
plt.ylabel('e(t), deg')
plt.title('Ошибка управления e(t)')
plt.legend()
plt.grid(True)

plt.show()
