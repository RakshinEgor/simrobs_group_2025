import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import numpy as np

f = "C:\\Users\\olivi\\Downloads\\cylinder.xml"

model = mujoco.MjModel.from_xml_path(f)
data = mujoco.MjData(model)

sim_time = 10
dt = model.opt.timestep
num_of_steps = int(sim_time / dt)
time_series = np.linspace(0, sim_time, num_of_steps)
O_position = []
t = 0.0

viewer = mujoco.viewer.launch_passive(model, data)

for i in range(num_of_steps):
    if not viewer.is_running():
        break
    data.ctrl[0] = 0.165 * np.sin(1 * np.pi * t) - 0.135
    t += dt
    O_position.append(data.sensordata[0])
    mujoco.mj_step(model, data)
    viewer.sync()
    
viewer.close()

plt.plot(time_series[800:], O_position[800:], '-', linewidth=2, label='P')
plt.grid()
plt.show()