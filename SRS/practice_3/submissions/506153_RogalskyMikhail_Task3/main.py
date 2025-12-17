import time

import mujoco
import mujoco.viewer

paused = False

def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused


m = mujoco.MjModel.from_xml_path('optimus.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:

  start = time.time()
  while viewer.is_running():
    step_start = time.time()


    if not paused:
        mujoco.mj_step(m, d)


        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        
        viewer.sync()
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)