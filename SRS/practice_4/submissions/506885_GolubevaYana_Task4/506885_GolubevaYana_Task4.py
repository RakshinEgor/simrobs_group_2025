import mujoco
import mujoco_viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("506885_GolubevaYana_Task3.xml")
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

print("PD-регулятор с двумя мышцами")

while viewer.is_alive:
    t = data.time
    
    desired_z = 0.75 + (16.85 * np.sin(1.9 * t) - 28.6) / 600.0
    
    # Текущая позиция
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "middle_attachment_00")
    current_z = data.site_xpos[site_id][2]
    current_vel = data.qvel[2]
    
    # PD-регулятор
    error = desired_z - current_z
    control = 5000 * error + 300 * (-current_vel)
    
    # Управление мышцами 
    data.ctrl[0] = (16.85 * np.sin(1.9 * t) - 28.6 + 60) / 120  # q1
    data.ctrl[1] = (16.09 * np.sin(3.53 * t) - 43.8 + 60) / 120  # q2
    
    data.ctrl[0] = max(0, min(1, data.ctrl[0]))  # ограничение
    data.ctrl[1] = max(0, min(1, data.ctrl[1]))  # ограничение
    
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()