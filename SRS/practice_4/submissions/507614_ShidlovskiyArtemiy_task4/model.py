import mujoco
import mujoco.viewer
import numpy as np
import os
import time

R1 = 0.012
R2 = 0.03
a = 0.062
b = 0.048
c = 0.056


def crossed_tangent_points(c1, r1, c2, r2):
    c1, c2 = np.asarray(c1), np.asarray(c2)
    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)
    if d < 1e-8:
        return None

    u = d_vec / d
    v = np.array([-u[1], u[0]])

    r2_eff = -r2
    denom = r1 - r2_eff
    cos_alpha = denom / d
    if abs(cos_alpha) > 1:
        return None
    sin_alpha = np.sqrt(1 - cos_alpha**2)

    pts = []
    for sign in [1, -1]:
        direction = cos_alpha*u + sign*sin_alpha*v
        p1 = c1 + r1*direction
        p2 = c2 + r2_eff*direction
        pts.append((p1, p2))
    return pts


def update_tendon_sites(model, data):
    block1_pos = data.body("body_block1").xpos[:2]
    block2_pos = data.body("body_block2").xpos[:2]

    tangents = crossed_tangent_points(block1_pos, R1, block2_pos, R2)
    if tangents is None:
        return

    tangents = sorted(tangents, key=lambda t: t[0][1], reverse=True)
    (t1_b1, t1_b2), (t2_b1, t2_b2) = tangents
    z = 0.0

    def global_to_body(pos, name):
        xpos = data.body(name).xpos
        xmat = data.body(name).xmat.reshape(3, 3)
        return xmat.T @ (np.array([pos[0], pos[1], z]) - xpos)

    # Перезаписываем позиции точек касания
    data.qpos[model.joint("t1_b1_x").qposadr[0]:model.joint("t1_b1_x").qposadr[0] + 2] = global_to_body(t1_b1,"body_block1")[:2]
    data.qpos[model.joint("t2_b1_x").qposadr[0]:model.joint("t2_b1_x").qposadr[0] + 2] = global_to_body(t2_b1,"body_block1")[:2]

    data.qpos[model.joint("t1_b2_x").qposadr[0]:model.joint("t1_b2_x").qposadr[0] + 2] = global_to_body(t1_b2,"body_block2")[:2]
    data.qpos[model.joint("t2_b2_x").qposadr[0]:model.joint("t2_b2_x").qposadr[0] + 2] = global_to_body(t2_b2,"body_block2")[:2]


def control_callback(model, data):

    # q1 (tendon1)
    AMP1 = 47.11
    FREQ1 = 3.27
    BIAS1 = 34.4

    # q2 (tendon2)
    AMP2  = 38.57
    FREQ2 = 1.29
    BIAS2 = -24.5

    q1_des = AMP1 * np.sin(FREQ1 * data.time) + BIAS1
    q2_des = AMP2 * np.sin(FREQ2 * data.time) + BIAS2

    tendon1_pos = data.sensordata[2]
    tendon2_pos = data.sensordata[3]

    tendon1_vel = data.sensordata[4]
    tendon2_vel = data.sensordata[5]

    Kp = 2.0
    Kd = 0.05

    tau1 = Kp * (q1_des - tendon1_pos) - Kd * tendon1_vel
    tau2 = Kp * (q2_des - tendon2_pos) - Kd * tendon2_vel

    data.ctrl[0] = tau1
    data.ctrl[1] = tau2

    update_tendon_sites(model, data)


if __name__ == "__main__":

    path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(path, "model2.xml")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    mujoco.set_mjcb_control(control_callback)

    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.001)

