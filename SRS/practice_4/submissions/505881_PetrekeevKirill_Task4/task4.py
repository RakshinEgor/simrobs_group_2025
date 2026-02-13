import mujoco
import mujoco.viewer
import time
import os
import numpy as np

def update_tendon_sites(model, data):

    block1_pos = data.body("body_block1").xpos[:2]
    block2_pos = data.body("body_block2").xpos[:2]
    
    block1_radius = model.geom("block1").size[0]
    block2_radius = model.geom("block2").size[0]
    
    tangents = calculate_internal_tangents(block1_pos, block1_radius, 
                                            block2_pos, block2_radius)
    
    z_pos = 0.0
    
    t1_block1_pos = tangents[0][0]
    t2_block1_pos = tangents[1][0]
    
    t1_block2_pos = tangents[0][1]
    t2_block2_pos = tangents[1][1]
    
    block1_xmat = data.body("body_block1").xmat.reshape(3, 3)
    block2_xmat = data.body("body_block2").xmat.reshape(3, 3)
    
    rel_t1_block1 = block1_xmat.T @ (np.array([t1_block1_pos[0], t1_block1_pos[1], z_pos]) - data.body("body_block1").xpos)
    rel_t2_block1 = block1_xmat.T @ (np.array([t2_block1_pos[0], t2_block1_pos[1], z_pos]) - data.body("body_block1").xpos)
    
    rel_t1_block2 = block2_xmat.T @ (np.array([t1_block2_pos[0], t1_block2_pos[1], z_pos]) - data.body("body_block2").xpos)
    rel_t2_block2 = block2_xmat.T @ (np.array([t2_block2_pos[0], t2_block2_pos[1], z_pos]) - data.body("body_block2").xpos)
    
    data.qpos[model.joint("t1_block1_joint").qposadr[0]:model.joint("t1_block1_joint").qposadr[0]+2] = rel_t1_block1[:2]
    data.qpos[model.joint("t2_block1_joint").qposadr[0]:model.joint("t2_block1_joint").qposadr[0]+2] = rel_t2_block1[:2]
    data.qpos[model.joint("t1_block2_joint").qposadr[0]:model.joint("t1_block2_joint").qposadr[0]+2] = rel_t1_block2[:2]
    data.qpos[model.joint("t2_block2_joint").qposadr[0]:model.joint("t2_block2_joint").qposadr[0]+2] = rel_t2_block2[:2]


def calculate_internal_tangents(center1, r1, center2, r2):
    
    d_vec = center2 - center1
    d = np.linalg.norm(d_vec)
    theta = np.arctan2(d_vec[1], d_vec[0])
    alpha = np.arcsin((r1 + r2) / d)
    
    angles1 = [theta + alpha, theta - alpha]
    angles2 = [theta + alpha + np.pi, theta - alpha + np.pi]
    
    points1 = []
    for angle in angles1:
        x = center1[0] + r1 * np.cos(angle)
        y = center1[1] + r1 * np.sin(angle)
        points1.append(np.array([x, y]))

    points2 = []
    for angle in angles2:
        x = center2[0] + r2 * np.cos(angle)
        y = center2[1] + r2 * np.sin(angle)
        points2.append(np.array([x, y]))
    
    return [(points1[0], points2[0]), (points1[1], points2[1])]



def callback(model, data):
    kd = 0.1
    q_des_1 = .1496 * np.sin(.0182 * data.time) - .0305
    q_des_2 = .1938 * np.sin(.0254 * data.time) + .0211
    # data.ctrl[0] = q_des_1
    # data.ctrl[1] = q_des_2
    kp = 0.1
    kd = 0.1
    
    q_curr_1 = data.sensor("block2_x_pos").data[0]
    q_curr_2 = data.sensor("block2_y_pos").data[0]
    qdot_curr_1 = data.sensor("block2_x_vel").data[0]
    qdot_curr_2 = data.sensor("block2_y_vel").data[0]
    
    data.ctrl[0] = kp * (q_des_1 - q_curr_1) - kd * qdot_curr_1
    data.ctrl[1] = kp * (q_des_2 - q_curr_2) - kd * qdot_curr_2
    
    update_tendon_sites(model, data)

# q_des_1 = 14.96 * np.sin(1.82 * data.time) - 30.5
# q_des_2 = 19.38 * np.sin(2.54 * data.time) + 21.1

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.xml")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

mujoco.set_mjcb_control(callback)

mujoco.viewer.launch(model, data)
# with mujoco.viewer.launch(model, data) as viewer:
#     while viewer.is_running():
#         viewer.sync()
#         time.sleep(0.01)