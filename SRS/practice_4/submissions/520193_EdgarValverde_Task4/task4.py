import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import numpy as np
import time

def deg2rad(deg): return deg * (np.pi / 180.0)

AMP1 = deg2rad(40.45)
FREQ1 = 2.93 
BIAS1 = deg2rad(-30.1)

AMP2 = deg2rad(26.52)
FREQ2 = 1.39
BIAS2 = deg2rad(-27.5)

Kp1 = 0.1
Kd1 = 0.001

Kp2 = 0.12
Kd2 = 0.01

model = mujoco.MjModel.from_xml_path("tendon.xml")
data = mujoco.MjData(model)

data.qpos[model.joint('joint1').qposadr[0]] = BIAS1
data.qpos[model.joint('joint2').qposadr[0]] = BIAS2

history = {"t": [], "q1_des": [], "q1_real": [], "q2_des": [], "q2_real": []}

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -30
    viewer.cam.distance = 0.4
    viewer.cam.lookat = np.array([0.05, 0.0, 0.1])

    time.sleep(2)

    while viewer.is_running():
        step_start = time.time()
        
        t = data.time
        
        omega1 = 2 * np.pi * FREQ1
        q1_des = AMP1 * np.sin(omega1 * t) + BIAS1
        v1_des = AMP1 * omega1 * np.cos(omega1 * t)
        
        omega2 = 2 * np.pi * FREQ2
        q2_des = AMP2 * np.sin(omega2 * t) + BIAS2
        v2_des = AMP2 * omega2 * np.cos(omega2 * t)
        
        q1_act = data.qpos[model.joint('joint1').qposadr[0]]
        v1_act = data.qvel[model.joint('joint1').dofadr[0]]
        
        q2_act = data.qpos[model.joint('joint2').qposadr[0]]
        v2_act = data.qvel[model.joint('joint2').dofadr[0]]
        
        tau1 = Kp1 * (q1_des - q1_act) + Kd1 * (v1_des - v1_act)
        tau2 = Kp2 * (q2_des - q2_act) + Kd2 * (v2_des - v2_act)
        
        data.ctrl[0] = tau1
        data.ctrl[1] = tau2
        
        mujoco.mj_step(model, data)
        
        history["t"].append(t)
        history["q1_des"].append(q1_des)
        history["q1_real"].append(q1_act)
        history["q2_des"].append(q2_des)
        history["q2_real"].append(q2_act)
        
        viewer.sync()
        
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
        if t > 5.0: break

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(history["t"], history["q1_des"], 'r--', label="q1 desired")
plt.plot(history["t"], history["q1_real"], 'r', label="q1 real")
plt.title("Joint 1")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(history["t"], history["q2_des"], 'b--', label="q2 desired")
plt.plot(history["t"], history["q2_real"], 'b', label="q2 real")
plt.title("Joint 2")
plt.xlabel("Time [s]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()