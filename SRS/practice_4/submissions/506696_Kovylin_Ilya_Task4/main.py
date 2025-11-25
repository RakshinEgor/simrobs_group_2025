import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import matplotlib.pyplot as plt


# Геометрия
def calculate_crossed_tangent_points(c1, r1, c2, r2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)
    if d < 1e-8:
        return None

    unit_d = d_vec / d
    perp_d = np.array([-unit_d[1], unit_d[0]])

    r2_eff = -r2
    cos_theta = (r1 - r2_eff) / d
    if abs(cos_theta) > 1.0:
        return None

    sin_theta = np.sqrt(1 - cos_theta**2)

    tangents = []
    for side in [1, -1]:
        dir_r = cos_theta * unit_d + side * sin_theta * perp_d
        p1 = c1 + r1 * dir_r
        p2 = c2 + r2_eff * dir_r
        tangents.append((p1, p2))

    return tangents



# Обновление положений

def update_tendon_sites(model, data):
    block1_pos = data.body("body_block1").xpos[:2]
    block2_pos = data.body("body_block2").xpos[:2]

    r1 = model.geom("block1").size[0]
    r2 = model.geom("block2").size[0]

    tangents = calculate_crossed_tangent_points(block1_pos, r1, block2_pos, r2)
    if tangents is None:
        return

    tangents = sorted(tangents, key=lambda t: t[0][1], reverse=True)
    (t1_b1, t1_b2), (t2_b1, t2_b2) = tangents

    z = 0.0
    b1_xmat = data.body("body_block1").xmat.reshape(3, 3)
    b2_xmat = data.body("body_block2").xmat.reshape(3, 3)

    rel_t1_b1 = np.dot(b1_xmat.T, [t1_b1[0], t1_b1[1], z] - data.body("body_block1").xpos)
    rel_t2_b1 = np.dot(b1_xmat.T, [t2_b1[0], t2_b1[1], z] - data.body("body_block1").xpos)
    rel_t1_b2 = np.dot(b2_xmat.T, [t1_b2[0], t1_b2[1], z] - data.body("body_block2").xpos)
    rel_t2_b2 = np.dot(b2_xmat.T, [t2_b2[0], t2_b2[1], z] - data.body("body_block2").xpos)

    data.qpos[model.joint("t1_block1_joint").qposadr[0]:
              model.joint("t1_block1_joint").qposadr[0] + 2] = rel_t1_b1[:2]

    data.qpos[model.joint("t2_block1_joint").qposadr[0]:
              model.joint("t2_block1_joint").qposadr[0] + 2] = rel_t2_b1[:2]

    data.qpos[model.joint("t1_block2_joint").qposadr[0]:
              model.joint("t1_block2_joint").qposadr[0] + 2] = rel_t1_b2[:2]

    data.qpos[model.joint("t2_block2_joint").qposadr[0]:
              model.joint("t2_block2_joint").qposadr[0] + 2] = rel_t2_b2[:2]



# Хранение траектории
time_log = []
q1_log, q1_des_log = [], []
q2_log, q2_des_log = [], []



# ПД 

def control_callback(model, data):

    k = 1

    KP1, KD1 = 135550.0 / k, 715.0 / k    
    KP2, KD2 = 135550.0 / k, 715.0 / k

    AMP1, FREQ1, BIAS1 = 18.7 / k, 3.8 / k, -36.7 / k
    AMP2, FREQ2, BIAS2 = 54.03 / k, 2.62 / k, 25.3 / k

    t = data.time

    # Ожидаемые
    q1_des = AMP1 * np.sin(FREQ1 * t) + BIAS1
    dq1_des = AMP1 * FREQ1 * np.cos(FREQ1 * t)

    q2_des = AMP2 * np.sin(FREQ2 * t) + BIAS2
    dq2_des = AMP2 * FREQ2 * np.cos(FREQ2 * t)

    # Текущие
    q1 = data.joint("block1_joint").qpos[0]
    dq1 = data.joint("block1_joint").qvel[0]

    q2 = data.joint("block2_joint").qpos[0]
    dq2 = data.joint("block2_joint").qvel[0]

    # ПД
    data.ctrl[0] = KP1 * (q1_des - q1) + KD1 * (dq1_des - dq1)
    data.ctrl[1] = KP2 * (q2_des - q2) + KD2 * (dq2_des - dq2)

    time_log.append(t)
    q1_log.append(q1)
    q1_des_log.append(q1_des)
    q2_log.append(q2)
    q2_des_log.append(q2_des)

    update_tendon_sites(model, data)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.xml")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

mujoco.set_mjcb_control(control_callback)

viewer = mujoco.viewer.launch(model, data)



# Отображение траектории
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_log, q1_log, label="q1 actual")
plt.plot(time_log, q1_des_log, '--', label="q1 desired")
plt.grid()
plt.legend()
plt.title("Block 1 trajectory")

plt.subplot(2, 1, 2)
plt.plot(time_log, q2_log, label="q2 actual")
plt.plot(time_log, q2_des_log, '--', label="q2 desired")
plt.grid()
plt.legend()
plt.title("Block 2 trajectory")

plt.tight_layout()
plt.show()



if viewer is None:
    raise RuntimeError("Не удалось запустить MuJoCo viewer. Возможно, проблема с GL или GLFW.")

while viewer.is_running():
    viewer.sync()
    time.sleep(0.002)

# окно закрыто пользователем → закрываем viewer корректно
viewer.close()


