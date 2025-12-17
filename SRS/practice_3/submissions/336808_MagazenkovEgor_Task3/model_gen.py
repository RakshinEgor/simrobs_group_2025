import mujoco
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np
import os
from lxml import etree
import mujoco.viewer
import time

def swap_par(tree, element_type, element_name, attribute_name, new_value):   
    # Находим тип и имя элемента, в котором меняем аттрибут
    # Find the type and name of the element in which we change the attribute
    
    element = tree.find(f'.//{element_type}[@name="{element_name}"]')
    
    # Меняем аттрибут
    # Change the attribute
    element.set(attribute_name, new_value)



f1 = "tendon.xml"
f2 = "tendon_mod.xml"

R_1 = 0.032
R_2 = 0.021
a = 0.054
b = 0.071
c = 0.055


tree = etree.parse(f1)

swap_par(tree, 'site', 'fix1', 'pos', f'0 0 {R_1/2}')
swap_par(tree, 'site', 'fix2', 'pos', f'0 0 {-R_1/2}')

swap_par(tree, 'geom', 'link 1 beam', 'pos', f'0.005 {a/2} 0')
swap_par(tree, 'geom', 'link 1 beam', 'size', f'0.001 {a/2}')

swap_par(tree, 'body', 'link 2', 'pos', f'0 {a} 0')
swap_par(tree, 'geom', 'circ 1', 'size', f'{R_1/2} 0 0')
swap_par(tree, 'site', 'fix1_1', 'pos', f'0 0 {R_1/2}')
swap_par(tree, 'site', 'fix2_1', 'pos', f'0 0 {-R_1/2}')

swap_par(tree, 'geom', 'link 2 beam', 'pos', f'0.005 {c/2} 0')
swap_par(tree, 'geom', 'link 2 beam', 'size', f'0.001 {c/2}')

swap_par(tree, 'body', 'link 3', 'pos', f'0 {c} 0')
swap_par(tree, 'geom', 'circ 2', 'size', f'{R_2/2} 0 0')
swap_par(tree, 'site', 'fix1_2', 'pos', f'0 0 {-R_2/2}')
swap_par(tree, 'site', 'fix2_2', 'pos', f'0 0 {R_2/2}')

swap_par(tree, 'geom', 'link 3 beam', 'pos', f'0.005 {b/2} 0')
swap_par(tree, 'geom', 'link 3 beam', 'size', f'0.001 {b/2}')

swap_par(tree, 'body', 'ee', 'pos', f'0 {b} 0')
swap_par(tree, 'site', 'rfix1', 'pos', f'0 0 {R_2/2}')
swap_par(tree, 'site', 'rfix2', 'pos', f'0 0 {-R_2/2}')

tree.write(f2, pretty_print=True, xml_declaration=True, encoding='UTF-8')
