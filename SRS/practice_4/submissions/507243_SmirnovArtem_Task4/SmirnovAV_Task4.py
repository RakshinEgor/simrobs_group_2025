import time
import mujoco
import mujoco.viewer
import numpy as np

paused = False

def key_callback(keycode):
    
        if chr(keycode) == ' ':
            global paused
            paused = not paused
          
m = mujoco.MjModel.from_xml_path('Task_4.xml')
d = mujoco.MjData(m)

def PD_contr(mj_data, KP, KV, theta):
  
    mj_data.ctrl[0] = KP * (-mj_data.qpos[0] + theta) + KV * (0 - mj_data.qvel[0])

TIME = 15
dt = 0.01
Ndt = int(TIME / dt)
t = np.linspace(0, TIME, Ndt)


FREQ = 3.86  
AMP = np.deg2rad(16.43) 
BIAS = np.deg2rad(2.2)  

theta_des = AMP * np.sin(FREQ * t) + BIAS




with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    start_time = time.time()
    
    for i in range(Ndt):
        if not viewer.is_running():
            break

        step_start = time.time()

        if not paused:
           
            PD_contr(d, 35, 12, theta_des[i])
            
            
            mujoco.mj_step(m, d)
            
            
            viewer.sync()

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)






