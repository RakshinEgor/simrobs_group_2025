import mujoco
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np
import os
from lxml import etree
import mujoco.viewer
import time
import re


R1 = 0.034
R2 = 0.044
a = 0.049
b = 0.059
c = 0.055

variables = {
    'R1': R1,
    'R2': R2,
    'a': a,
    'b': b,
    'c': c
}

f1 = "mujoco3.xml"
f2 = "mujoco3_1.xml"


with open(f1, 'rb') as f:
    xml_bytes = f.read()

xml_str = xml_bytes.decode('utf-8')

def evaluate_expression(match):
    expression = match.group(1)  
    try:
        result = eval(expression, {}, variables)
        return str(result)
    except Exception as e:
        print(f"Ошибка при вычислении выражения {expression}: {e}")
        return match.group(0)  

pattern = r'\{([^}]+)\}'

xml_str_modified = re.sub(pattern, evaluate_expression, xml_str)
xml_bytes_modified = xml_str_modified.encode('utf-8')
tree = etree.fromstring(xml_bytes_modified)

etree.ElementTree(tree).write(f2, pretty_print=True, xml_declaration=True, encoding='UTF-8')

model = mujoco.MjModel.from_xml_path(f2)
data = mujoco.MjData(model)

SIMEND = 20
TIMESTEP = 0.001
STEP_NUM = int(SIMEND / TIMESTEP)
timeseries = np.linspace(0, SIMEND, STEP_NUM)

viewer = mujoco_viewer.MujocoViewer(model, data, title="tendon", width=1920, height=1080)

while viewer.is_alive: 
        
        mujoco.mj_step(model, data)
        viewer.render()


viewer.close()

