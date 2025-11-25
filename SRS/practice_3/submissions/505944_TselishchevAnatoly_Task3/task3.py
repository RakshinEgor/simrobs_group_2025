import mujoco
import mujoco.viewer

model_path = "task3.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
