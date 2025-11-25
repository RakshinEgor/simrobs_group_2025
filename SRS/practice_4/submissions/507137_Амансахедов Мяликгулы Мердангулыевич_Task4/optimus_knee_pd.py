import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

AMP_deg = 21.06
FREQ_hz = 1.87
BIAS_deg = 38.5

AMP = np.deg2rad(AMP_deg)
BIAS = np.deg2rad(BIAS_deg)
omega = 2 * np.pi * FREQ_hz

Kp = 80.0
Kd = 2.0

SIM_TIME = 10.0

xml_path = "4bar.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

joint_name = "A"
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
if joint_id < 0:
    raise ValueError(f"Joint '{joint_name}' not found in model")

q_adr = model.jnt_qposadr[joint_id]
qd_adr = model.jnt_dofadr[joint_id]

if model.nu < 1:
    raise ValueError("No actuators in the model. Добавь <motor ...> в XML.")
actuator_id = 0

time_log = []
q_log = []
q_des_log = []
tau_log = []
with mujoco.viewer.launch_passive(model, data) as viewer:

    mujoco.mj_forward(model, data)

    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "sP")
        if site_id >= 0:
            viewer.cam.lookat[:] = data.site_xpos[site_id]
    except Exception:
        pass

    viewer.cam.distance = 0.4
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90

    while viewer.is_running() and data.time < SIM_TIME:
        t = data.time

        q_des = AMP * np.sin(omega * t) + BIAS
        qd_des = AMP * omega * np.cos(omega * t)

        q = float(data.qpos[q_adr])
        qd = float(data.qvel[qd_adr])

        tau = Kp * (q_des - q) + Kd * (qd_des - qd)

        data.ctrl[actuator_id] = tau

        time_log.append(t)
        q_log.append(q)
        q_des_log.append(q_des)
        tau_log.append(tau)

        mujoco.mj_step(model, data)
        viewer.sync()

print("Simulation finished, building plots...")

time_log = np.array(time_log)
q_log = np.array(q_log)
q_des_log = np.array(q_des_log)
tau_log = np.array(tau_log)

plt.figure(figsize=(12, 5))
plt.plot(time_log, q_des_log, label="q желаемый", linewidth=2)
plt.plot(time_log, q_log, label="q фактический", linewidth=2, alpha=0.8)
plt.grid(True)
plt.title("Отработка PD-регулятора для сустава A")
plt.xlabel("Время [с]")
plt.ylabel("Угол [рад]")
plt.legend()
plt.tight_layout()
plt.savefig("pd_tracking.png", dpi=300)

plt.figure(figsize=(12, 5))
plt.plot(time_log, tau_log, color="orange", linewidth=1.5)
plt.grid(True)
plt.title("Сигнал управления τ(t)")
plt.xlabel("Время [с]")
plt.ylabel("Момент τ [Н·м]")
plt.tight_layout()
plt.savefig("pd_control.png", dpi=300)

plt.show()

err = q_des_log - q_log
print(f"Максимальная ошибка: {np.max(np.abs(err)):.4f} рад")
print(f"RMS ошибка:         {np.sqrt(np.mean(err**2)):.4f} рад")
print(f"MAE ошибка:         {np.mean(np.abs(err)):.4f} рад")
