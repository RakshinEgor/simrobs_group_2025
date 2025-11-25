import mujoco
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
import time

f1 = "C:\\Users\\yanan\\.vscode\\506885_GolubevaYana_Task3.xml"
f2 = "C:\\Users\\yanan\\.vscode\\506885_GolubevaYana_Task3_modified.xml"

def swap_par(tree, element_type, element_name, attribute_name, new_value):   
    element = tree.find(f'.//{element_type}[@name="{element_name}"]')
    if element is not None:
        element.set(attribute_name, new_value)

a = 0.076
b = 0.098
c = 0.049
R1 = 0.035
R2 = 0.032
base_height = 0.75

tree = etree.parse(f1)

# Настройка модели
swap_par(tree, 'body', 'R1', 'pos', f"0 {a} {base_height}")
swap_par(tree, 'body', 'R2', 'pos', f"0 {b} {base_height}")
swap_par(tree, 'body', 'S', 'pos', f"0 {c} {base_height}")

swap_par(tree, 'geom', 'R1_geom', 'size', f"{R1}")
swap_par(tree, 'geom', 'R2_geom', 'size', f"{R2}")

swap_par(tree, 'site', 'top_attachment_1', 'pos', f"0 0 {R1}")
swap_par(tree, 'site', 'middle_attachment_1', 'pos', "0 0 0")
swap_par(tree, 'site', 'bottom_attachment_1', 'pos', f"0 0 -{R1}")

swap_par(tree, 'site', 'top_attachment_2', 'pos', f"0 0 {R2}")
swap_par(tree, 'site', 'middle_attachment_2', 'pos', "0 0 0")
swap_par(tree, 'site', 'bottom_attachment_2', 'pos', f"0 0 -{R2}")

tree.write(f2, pretty_print=True, xml_declaration=True, encoding='UTF-8')

# Загрузка модели
model = mujoco.MjModel.from_xml_path(f2)
data = mujoco.MjData(model)

# Функция управления
def set_torque(mj_data, time_val, amplitude, frequency, phase):
    if model.nu >= 2:
        data.ctrl[0] = amplitude * np.sin(time_val * frequency + phase)
        data.ctrl[1] = -data.ctrl[0]
    elif model.nu >= 1:
        data.ctrl[0] = amplitude * np.sin(time_val * frequency + phase)

# Параметры симуляции
SIMEND = 20
TIMESTEP = 0.001
STEP_NUM = int(SIMEND / TIMESTEP)

A = 3
F = 30
P = 0

ee_pos_x = []
ee_pos_z = []

for i in range(STEP_NUM):
    set_torque(data, data.time, A, F, P)
    
    site_id = 8  
    position_EE = data.site_xpos[site_id]
    ee_pos_x.append(position_EE[0])
    ee_pos_z.append(position_EE[2])
    
    mujoco.mj_step(model, data)
if len(ee_pos_x) > 0:
    midlength = int(len(ee_pos_x)/2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ee_pos_x, ee_pos_z, '-', linewidth=2, label='End Effector')
    plt.title('End-effector trajectory', fontsize=12, fontweight='bold')
    plt.legend(loc='upper left')
    plt.xlabel('X-Axis [m]')
    plt.ylabel('Z-Axis [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    base_height_z = 1.0
    error = sum((np.array(ee_pos_z[midlength:]) - base_height_z * np.ones(len(ee_pos_z[midlength:]))) ** 2)/len(ee_pos_z[midlength:])
    print(f"Error: {error}")
    