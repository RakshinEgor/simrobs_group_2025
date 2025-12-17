import mujoco
import mujoco.viewer
import time
import sys

# Define constant speeds for the tendon actuators
PULLEY_SPEED_1 = 0.06
PULLEY_SPEED_2 = -0.01
# Define geometric constants for the 2R mechanism and pulleys
# R1, R2: Diameter of the two pulleys
# a, b, c: Lengths of the linkage segments
R1, R2, a, b, c = 0.019, 0.012, 0.052, 0.044, 0.052

def generate_model_xml(R1: float, R2: float, a: float, b: float, c: float):    
    return f"""
        <mujoco model="2R_tendon_planar">

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.5 0.5 0.5" width="265" height="256"/>
        </asset>

        <option timestep="1e-4" integrator="RK4" gravity="0 0 -9.81"/>

        <worldbody>

            <body name="wall" pos="0 0 0" euler="0 90 0">
                <geom type="box" size="0.05 0.05 0.005" pos="0 0 0" rgba="0.5 0.5 0.5 1"/>
                <site name="start_green" pos="{R1 / 2} 0 0" type="sphere" size="0.001" rgba="1 0 0 0.5" />
                <site name="start_red" pos="{-R1 / 2} 0 0" type="sphere" size="0.001" rgba="0 1 0 0.5" />
            </body>

            <body name="green_mid" pos="{a + c / 2} 0 0">
                <site name="mid_green" pos="0 0 0" type="sphere" size="0.001"/>
                <joint name="mid_joint_x_green" type="slide" axis="1 0 0"/>
                <joint name="mid_joint_y_green" type="slide" axis="0 0 1"/>
                <geom type="sphere" size="0.001" mass="0.0001" rgba="0 1 0 0.5" contype="0"/>
            </body>

            <body name="red_mid" pos="{a + c / 2} 0 0">
                <site name="mid_red" pos="0 0 0" type="sphere" size="0.001"/>
                <joint name="mid_joint_x_red" type="slide" axis="1 0 0"/>
                <joint name="mid_joint_y_red" type="slide" axis="0 0 1"/>
                <geom type="sphere" size="0.001" mass="0.0001" rgba="1 0 0 0.5" contype="0"/>
            </body>

            <body name="link_end" pos="{a + b + c} 0 0">
                <site name="end_link" pos="0 0 0" type="sphere" size="0.001"/>
                <joint name="end_x" type="slide" axis="1 0 0"/>
                <joint name="end_y" type="slide" axis="0 0 1"/>
                <geom type="box" size="0.002 0.002 {R2 / 2}" rgba="1 1 0 0.5" mass="0.001" contype="0"/>
            </body>

            <body name="link_1" pos="0 0 0">         
                <geom type="cylinder" pos="{a / 2} 0 0" size="0.0002 {a / 2}" euler="0 90 0" rgba="0 1 1 0.5" contype="0"/>

                <body name="link_2" pos="{a} 0 0">
                    <joint name="rotate_1" type="hinge" axis="0 1 0" stiffness="0" springref="0" damping="0"/>     
                    <geom type="cylinder" pos="{c / 2} 0 0" size="0.0002 {c / 2}" euler="0 90 0" rgba="0 1 1 0.5" contype="0"/>

                    <geom name="pulley_r1" type="cylinder" size="{R1 / 2} 0.01" pos="0 0 0" euler="90 0 0" rgba="1 0 0 0.5" contype="0"/>
                    <site name="side_r1_green" pos="0 0 {-R1 / 2 - 0.0001}" type="sphere" size="0.001"/>
                    <site name="side_r1_red" pos="0 0 {R1 / 2 + 0.0001}" type="sphere" size="0.001"/>
                    <site name="pulley_r1_center" pos="0 0 0" type="sphere" size="0.001"/>

                    <body name="link_3" pos="{c} 0 0">
                        <joint name="rotate_2" type="hinge" axis="0 1 0" stiffness="0" springref="0" damping="0"/>     
                        <geom type="cylinder" pos="{b / 2} 0 0" size="0.0002 {b / 2}" euler="0 90 0" rgba="0 1 1 0.5" contype="0"/>

                        <site name="end_green" pos="{b} 0 {R2 / 2}" type="sphere" size="0.001"/>
                        <site name="end_red" pos="{b} 0 {-R2 / 2}" type="sphere" size="0.001"/>

                        <geom name="pulley_r2" type="cylinder" size="{R2 / 2} 0.01" pos="0 0 0" euler="90 0 0" rgba="0 1 0 0.5" contype="0"/>
                        <site name="side_r2_green" pos="0 0 {R2 / 2 + 0.0001}" type="sphere" size="0.001"/>
                        <site name="side_r2_red" pos="0 0 {-R2 / 2 - 0.0001}" type="sphere" size="0.001"/>
                        <site name="pulley_r2_center" pos="0 0 0" type="sphere" size="0.001"/>

                        <site name="end" pos="{b} 0 0" type="sphere" size="0.001"/>
                    </body>
                </body>
            </body>
        </worldbody>

        <tendon>
            <spatial name="tendon_green" width="0.001" rgba="1 0 0 0.5" stiffness="1" damping="10" springlength="0.005">
                <site site="start_green"/>
                <geom geom="pulley_r1" sidesite="side_r1_green"/>
                <site site="mid_green"/>
                <geom geom="pulley_r2" sidesite="side_r2_green"/>
                <site site="end_green"/>
            </spatial>
       
            <spatial name="tendon_red" width="0.001" rgba="0 1 0 0.5" stiffness="1" damping="10" springlength="0.005">
                <site site="start_red"/>
                <geom geom="pulley_r1" sidesite="side_r1_red"/>
                <site site="mid_red"/>
                <geom geom="pulley_r2" sidesite="side_r2_red"/>
                <site site="end_red"/>
            </spatial>
        </tendon>

        <equality>
            <weld site1="end" site2="end_link" torquescale="100"/>

            <connect site1="mid_green" site2="pulley_r1_center"/>
            <connect site1="mid_green" site2="pulley_r2_center"/>
            <connect site1="mid_red" site2="pulley_r1_center"/>
            <connect site1="mid_red" site2="pulley_r2_center"/>
        </equality>

        <actuator>
            <velocity name="motor_green" tendon="tendon_green" gear="1" ctrlrange="-1 1"/>
            <velocity name="motor_red" tendon="tendon_red" gear="1" ctrlrange="-1 1"/>
        </actuator>

    </mujoco>
    """

def main():
    # Generate the XML model
    xml = generate_model_xml(R1, R2, a, b, c)
    # Load the model and create the data structure
    model = mujoco.MjModel.from_xml_string(xml.encode("utf-8"))
    data = mujoco.MjData(model)
    # Set the test control signals for the actuators
    data.ctrl[0] = PULLEY_SPEED_1
    data.ctrl[1] = PULLEY_SPEED_2

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            
            while viewer.is_running():
                # Simulation step loop
                simstart = data.time
                while data.time - simstart < 1.0/60.0:
                    mujoco.mj_step(model, data)
                # Update the viewer's scene
                viewer.sync()
                # Sleep to maintain the real-time simulation rate
                time.sleep(model.opt.timestep - data.time % model.opt.timestep)

    except Exception as e:
        # Handle potential errors in the simulation process
        print(f"Error during simulation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()