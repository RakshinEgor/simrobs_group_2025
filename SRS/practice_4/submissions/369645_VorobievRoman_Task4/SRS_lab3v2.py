from math import atan2, pi
from re import A
import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

paused = False

def main(): 

    # «агружаем xml модель
    f = 'lab3.xml'
    m = mujoco.MjModel.from_xml_path(f)
    d = mujoco.MjData(m)

    AMP = np.deg2rad(32.7)
    FREQ = 3.56
    BIAS = np.deg2rad(-25.3)

    KP = 750
    KD = 1e-7

    # site конечного звена
    q1_des = []
    q1 = []
    error = []
    t = []
    err_prev = BIAS

    t0 = time.time()
    t_prev = t0

    # запуск viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if not paused:
                mujoco.mj_step(m, d)

                viewer.sync()
                theta1_des = AMP*np.sin(FREQ*(time.time() - t0)) + BIAS
                theta1 = d.qpos[0]

                err = theta1_des - theta1

                dT = time.time() - t_prev
                derr = (err - err_prev)/dT

                d.ctrl[0] = KP * err + KD*derr 
                t_prev = time.time()

                if time.time()-t0 > 2:              
                    q1.append(theta1)
                    q1_des.append(theta1_des)
                    error.append(err)
                    t.append(time.time() - t0)

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(t, q1, linewidth=2)
    ax.plot(t, q1_des, linewidth=2, color='orange')
    ax.set_title("Angle")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("q1 [deg]")
    ax.grid(True)

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.plot(t, error, linewidth=2)
    ax.set_title("Control error")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("dq1 [deg]")
    ax.grid(True)
    
    plt.show()
    input("\n ")

if __name__ == "__main__":
    main()