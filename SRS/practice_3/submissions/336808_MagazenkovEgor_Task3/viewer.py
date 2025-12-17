import mujoco
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np
import os
from lxml import etree
import mujoco.viewer
import time


f2 = "tendon.xml"

model = mujoco.MjModel.from_xml_path(f2)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, 
                                    data, 
                                    title="4bar", 
                                    width=1920, 
                                    height=1080)

while True:
    mujoco.mj_step(model, data)
    viewer.render()
    
viewer.close()
