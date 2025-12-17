import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

from tendon import generate_tendon_xml   # your XML generator


# -----------------------------
# 1. Parameters of your variant
# -----------------------------
R1 = 0.013   # left pulley radius
R2 = 0.045   # right pulley radius
a  = 0.040   # distance to first joint
b  = 0.042   # distance from second joint to effector
c  = 0.087   # distance between pulleys


# -----------------------------
# 2. Build model and data
# -----------------------------
xml_string = generate_tendon_xml(R1, R2, a, b, c)

# create MuJoCo model from XML string
model = mujoco.MjModel.from_xml_string(xml_string)
data  = mujoco.MjData(model)

# small initial rotation so we see some motion (passive)
if model.nq >= 1:
    data.qpos[0] = 0.3   # joint A ~ 17 degrees
if model.nq >= 2:
    data.qpos[1] = 0.0   # joint B

mujoco.mj_forward(model, data)


# -----------------------------
# 3. Get effector site id
#    (we read its position directly, no sensors needed)
# -----------------------------
eff_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "effector")


# -----------------------------
# 4. Simulation settings
# -----------------------------
SIMEND   = 20.0               # simulate 20 seconds of model time
dt       = model.opt.timestep # = 1e-4 from your XML
STEP_NUM = int(SIMEND / dt)

# arrays to log end-effector trajectory
eff_x = []
eff_z = []


# -----------------------------
# 5. Run passive simulation with viewer
# -----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(STEP_NUM):
        if not viewer.is_running():
            break

        # PASSIVE DYNAMICS ONLY:
        # no data.ctrl, no motors, no external forces

        # log effector position (world coordinates)
        pos = data.site_xpos[eff_id]   # [x, y, z]
        eff_x.append(pos[0])
        eff_z.append(pos[2])

        # one simulation step
        mujoco.mj_step(model, data)

        # update viewer
        viewer.sync()


# -----------------------------
# 6. Plot end-effector trajectory
# -----------------------------
plt.figure()
plt.plot(eff_x, eff_z, '-', linewidth=2, label='End-effector')
plt.title('End-effector trajectory (passive model)', fontsize=12, fontweight='bold')
plt.xlabel('X-axis [m]')
plt.ylabel('Z-axis [m]')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
