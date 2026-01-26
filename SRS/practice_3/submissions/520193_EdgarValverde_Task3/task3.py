import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import numpy as np
import time
import glfw

NSTEPS = 2000
SITE_NAME = "end_effector"

model = mujoco.MjModel.from_xml_path("tendon.xml")
data = mujoco.MjData(model)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)

print(f"  - DOFs: {model.nq}")
print(f"  - Number of bodys: {model.nbody}")
print(f"  - Number of joints: {model.njnt}")
print(f"  - Number of actuators: {model.nu}")

positions_x = []
positions_z = []
times = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -30
    viewer.cam.distance = 0.4
    viewer.cam.lookat = np.array([0.05, 0.0, 0.1])

    time.sleep(2)

    for i in range(NSTEPS):
        if viewer.is_running():
            t = data.time
            data.ctrl[0] = 0.01*np.sin(50 * t)
            data.ctrl[1] = -0.01*np.cos(50 * t)

            pos = data.site_xpos[site_id].copy()
            positions_x.append(pos[0])
            positions_z.append(pos[2])
            times.append(t)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    viewer.close()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(positions_x, positions_z, 'b-', linewidth=1)
plt.scatter(positions_x[0], positions_z[0], color='green', s=50, label='Start')
plt.scatter(positions_x[-1], positions_z[-1], color='red', s=50, label='End')
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title(f'Trajectory of end effector')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Subplot 2: Position vs Time
plt.subplot(1, 3, 2)
plt.plot(times, positions_x, label='X')
plt.plot(times, positions_z, label='Z')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title('Position vs Time')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: 3D Trajectory
ax = plt.subplot(1, 3, 3, projection='3d')
ax.plot(positions_x, positions_z, times, 'b-', alpha=0.7)
ax.scatter(positions_x[0], positions_z[0], times[0], color='green', s=50)
ax.scatter(positions_x[-1], positions_z[-1], times[-1], color='red', s=50)
ax.set_xlabel('X [m]')
ax.set_ylabel('Z [m]')
ax.set_zlabel('Time [s]')
ax.set_title('3D Trajectory')

plt.show()




