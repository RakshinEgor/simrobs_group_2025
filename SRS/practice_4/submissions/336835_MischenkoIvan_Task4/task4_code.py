import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path(
    "SRS\\practice_4\\submissions\\336835_MischenkoIvan_Task4\\task4_model.xml")
data = mujoco.MjData(model)

motor1_id = model.actuator('motor_A').id # q1
motor2_id = model.actuator('motor_B').id # q2

joint_q1 = model.joint('A').id
joint_q2 = model.joint('B').id

joint_q1_pos = model.jnt_qposadr[joint_q1]
joint_q2_pos = model.jnt_qposadr[joint_q2]

joint_q1_vel = model.jnt_dofadr[joint_q1]
joint_q2_vel = model.jnt_dofadr[joint_q2]

# Расчет значений параметров
deg2rad = np.pi / 180

AMP1 = 24.1 * deg2rad
FREQ1 = 3.57
BIAS1 = 40 * deg2rad

AMP2 = 28.61 * deg2rad
FREQ2 = 1.48
BIAS2 = 7 * deg2rad

Kp1 = 8.5
Kd1 = 0.025
Kp2 = 4.5
Kd2 = 0.025

ctrl_min = model.actuator_ctrlrange[:, 0]
ctrl_max = model.actuator_ctrlrange[:, 1]

dt = model.opt.timestep
sim_time = 5.0
steps = int(sim_time / dt)
time = np.arange(steps) * dt
t0 = 0.0

q1_start = AMP1 * np.sin(2 * np.pi * FREQ1 * t0) + BIAS1
q2_start = AMP2 * np.sin(2 * np.pi * FREQ2 * t0) + BIAS2

data.qpos[joint_q1_pos] = -q1_start
data.qpos[joint_q2_pos] = -q2_start
data.qvel[joint_q1_vel] = 0
data.qvel[joint_q2_vel] = 0

mujoco.mj_forward(model, data)

q1_log = []
q2_log = []
q1_des_log = []
q2_des_log = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 0.5
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20

    for i in range(steps):
        t = i * dt

        q1_pos_des = AMP1 * np.sin(2 * np.pi * FREQ1 * t) + BIAS1
        q2_pos_des = AMP2 * np.sin(2 * np.pi * FREQ2 * t) + BIAS2
        q1_vel_des = AMP1 * 2 * np.pi * FREQ1 * np.cos(2 * np.pi * FREQ1 * t)
        q2_vel_des = AMP2 * 2 * np.pi * FREQ2 * np.cos(2 * np.pi * FREQ2 * t)

        q1_pos = data.qpos[joint_q1_pos]
        q2_pos = data.qpos[joint_q2_pos]
        q1_vel = data.qvel[joint_q1_vel]
        q2_vel = data.qvel[joint_q2_vel]

        u1 = Kp1 * (q1_pos_des - q1_pos) + Kd1 * (q1_vel_des - q1_vel)
        u2 = Kp2 * (q2_pos_des - q2_pos) + Kd2 * (q2_vel_des - q2_vel)

        u1 = float(np.clip(u1, ctrl_min[motor1_id], ctrl_max[motor1_id]))
        u2 = float(np.clip(u2, ctrl_min[motor2_id], ctrl_max[motor2_id]))

        data.ctrl[motor1_id] = u1
        data.ctrl[motor2_id] = u2

        mujoco.mj_step(model, data)
        viewer.sync()

        q1_log.append(q1_pos)
        q2_log.append(q2_pos)
        q1_des_log.append(q1_pos_des)
        q2_des_log.append(q2_pos_des)

plt.figure(figsize=(10, 6))
plt.plot(time, q1_log, label="q1")
plt.plot(time, q1_des_log, label="q1_des")
plt.title("Joint 1 trajectory")
plt.legend()
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, q2_log, label="q2")
plt.plot(time, q2_des_log, label="q2_des")
plt.title("Joint 2 trajectory")
plt.legend()
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.show()
