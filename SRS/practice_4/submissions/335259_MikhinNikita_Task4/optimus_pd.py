import os
import math
import numpy as np

import mujoco
import mujoco_viewer
import matplotlib.pyplot as plt

AMP_DEG = 39.6
FREQ_HZ = 1.29
BIAS_DEG = 30.4

AMP = math.radians(AMP_DEG)
BIAS = math.radians(BIAS_DEG)
OMEGA = 2.0 * math.pi * FREQ_HZ

KP = 25.0
KD = 2.0

REF_RAMP_TAU = 0.8

VEL_ERR_MAX = 3.0


def make_controller(model: mujoco.MjModel, data: mujoco.MjData):
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "J_A")
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "q1_motor")
    if jnt_id < 0 or act_id < 0:
        raise RuntimeError("Failed to resolve joint/actuator IDs (J_A / q1_motor)")

    qpos_addr = model.jnt_qposadr[jnt_id]
    qvel_addr = model.jnt_dofadr[jnt_id]

    ctrl_min = float(model.actuator_ctrlrange[act_id, 0]) if model.actuator_ctrllimited[act_id] else -np.inf
    ctrl_max = float(model.actuator_ctrlrange[act_id, 1]) if model.actuator_ctrllimited[act_id] else np.inf

    u_prev = 0.0
    alpha_u = 0.3

    def controller(_model, _data):
        nonlocal u_prev
        t = float(_data.time)

        amp_scale = 1.0 - math.exp(-t / REF_RAMP_TAU)
        q_des = (AMP * amp_scale) * math.sin(OMEGA * t) + BIAS
        dq_des = (AMP * amp_scale) * OMEGA * math.cos(OMEGA * t)

        q = float(_data.qpos[qpos_addr])
        dq = float(_data.qvel[qvel_addr])

        v_err = max(min(dq_des - dq, VEL_ERR_MAX), -VEL_ERR_MAX)
        u_raw = KP * (q_des - q) + KD * v_err

        u = alpha_u * u_raw + (1.0 - alpha_u) * u_prev
        u = max(min(u, ctrl_max), ctrl_min)
        u_prev = u

        u = max(min(u, ctrl_max), ctrl_min)

        _data.ctrl[act_id] = u

    return controller


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(here, "optimus.xml")

    print(f"Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    mujoco.set_mjcb_control(make_controller(model, data))

    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "J_A")
    qpos_addr = model.jnt_qposadr[jnt_id]
    qvel_addr = model.jnt_dofadr[jnt_id]

    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "q1_motor")
    if act_id < 0:
        raise RuntimeError("Actuator q1_motor not found")
    ctrl_min = float(model.actuator_ctrlrange[act_id, 0]) if model.actuator_ctrllimited[act_id] else -np.inf
    ctrl_max = float(model.actuator_ctrlrange[act_id, 1]) if model.actuator_ctrllimited[act_id] else np.inf

    times = []
    q_traj = []
    dq_traj = []
    qdes_traj = []
    dqdes_traj = []
    err_traj = []
    u_traj = []
    sat_traj = []
    LOG_DECIMATE = 20
    step_count = 0

    title = "Practice 4 — Optimus knee q1 PD"
    viewer = mujoco_viewer.MujocoViewer(model, data, title=title)
    try:
        print("Running simulation. Press ESC in the viewer to quit.")
        STEPS_PER_RENDER = 10
        while True:
            alive_attr = getattr(viewer, "is_alive")
            alive = alive_attr if isinstance(alive_attr, bool) else alive_attr()
            if not alive:
                break
            for _ in range(STEPS_PER_RENDER):
                mujoco.mj_step(model, data)
                step_count += 1
                if step_count % LOG_DECIMATE == 0:
                    t = float(data.time)
                    q = float(data.qpos[qpos_addr])
                    dq = float(data.qvel[qvel_addr])
                    q_des = AMP * math.sin(OMEGA * t) + BIAS
                    dq_des = AMP * OMEGA * math.cos(OMEGA * t)
                    e = q_des - q
                    v_err = max(min(dq_des - dq, VEL_ERR_MAX), -VEL_ERR_MAX)
                    u = KP * e + KD * v_err
                    saturated = False
                    if u > ctrl_max:
                        u = ctrl_max
                        saturated = True
                    elif u < ctrl_min:
                        u = ctrl_min
                        saturated = True

                    times.append(t)
                    q_traj.append(q)
                    dq_traj.append(dq)
                    qdes_traj.append(q_des)
                    dqdes_traj.append(dq_des)
                    err_traj.append(e)
                    u_traj.append(u)
                    sat_traj.append(1 if saturated else 0)
            viewer.render()
    finally:
        viewer.close()

    if len(times) > 1:
        here = os.path.dirname(os.path.abspath(__file__))
        png_path = os.path.join(here, "timeseries.png")
        csv_path = os.path.join(here, "timeseries.csv")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax1.plot(times, q_traj, label="q (rad)")
        ax1.plot(times, qdes_traj, '--', label="q_des (rad)")
        ax1.set_ylabel("angle, rad")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(times, err_traj, color='C3', label="error = q_des - q")
        ax2.set_xlabel("time, s")
        ax2.set_ylabel("error, rad")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.suptitle("Optimus knee q1: angle and error")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

        import numpy as np
        arr = np.column_stack([times, q_traj, dq_traj, qdes_traj, dqdes_traj, err_traj, u_traj, sat_traj])
        np.savetxt(
            csv_path,
            arr,
            delimiter=",",
            header="time,q,dq,q_des,dq_des,error,u,sat",
            comments="",
        )
        print(f"Saved plot: {png_path}")
        print(f"Saved data: {csv_path}")
    else:
        print("No samples collected for plotting (simulation too short).")


if __name__ == "__main__":
    main()
