import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from lxml import etree

L_AB = 0.036
L_BC = 0.0468
L_CD = 0.054
L_AD = 0.036
L_CP = 0.18


def swap_attr(tree, tag, name, attr, value):
    elem = tree.find(f".//{tag}[@name='{name}']")
    if elem is None:
        raise ValueError(f"Element <{tag} name='{name}'> not found")
    elem.set(attr, value)


base_xml = "4bar.xml"
out_xml = "optimus_knee.xml"

tree = etree.parse(base_xml)

swap_attr(tree, "geom", "link AB", "pos", f"0 -{L_AB/2:.5f} 0")
swap_attr(tree, "geom", "link AB", "size", f"0.015 {L_AB/2:.5f}")

swap_attr(tree, "body", "BC1P", "pos", f"0 -{L_AB:.5f} 0")

swap_attr(tree, "geom", "link BC", "pos", f"0 -{L_BC/2:.5f} 0")
swap_attr(tree, "geom", "link BC", "size", f"0.015 {L_BC/2:.5f}")

swap_attr(tree, "site", "sC1", "pos", f"0 -{(L_AB + L_BC):.5f} 0")

swap_attr(tree, "body", "CP", "pos", f"0 -{(L_AB + L_BC):.5f} 0")

swap_attr(tree, "geom", "link CP", "pos", f"0 -{L_CP/2:.5f} 0")
swap_attr(tree, "geom", "link CP", "size", f"0.015 {L_CP/2:.5f}")

swap_attr(tree, "site", "sP", "pos", f"0 -{L_CP:.5f} 0")

swap_attr(tree, "body", "DC2", "pos", f"{L_AD:.5f} 0 1.5")

swap_attr(tree, "geom", "link DC", "pos", f"0 -{L_CD/2:.5f} 0")
swap_attr(tree, "geom", "link DC", "size", f"0.015 {L_CD/2:.5f}")

swap_attr(tree, "site", "sC2", "pos", f"0 -{L_CD:.5f} 0")

act_elem = tree.find(".//actuator/position")
if act_elem is not None:
    act_elem.set("joint", "D")

tree.write(out_xml, pretty_print=True, xml_declaration=True, encoding="UTF-8")
print(f"XML with Optimus parameters saved to: {out_xml}")

model = mujoco.MjModel.from_xml_path(out_xml)
data = mujoco.MjData(model)

P_traj_x = []
P_traj_z = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "sP")
    viewer.cam.lookat[:] = data.site_xpos[site_id]

    viewer.cam.distance = 0.3
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90

    freq = 1.0
    amp = 0.7

    while viewer.is_running():
        data.ctrl[0] = amp * np.sin(2 * np.pi * freq * data.time)
        mujoco.mj_step(model, data)
        viewer.sync()
    freq = 1.0  # Гц
    amp = 0.7  # рад

    while viewer.is_running():
        data.ctrl[0] = amp * np.sin(2 * np.pi * freq * data.time)
        mujoco.mj_step(model, data)
        viewer.sync()

print("Simulation finished.")
