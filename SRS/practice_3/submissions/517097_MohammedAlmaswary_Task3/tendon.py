def generate_tendon_xml(R1: float, R2: float, a: float, b: float, c: float):
    stiffness = 100
    
    return f"""
<mujoco model="2R_tendon_planar">

    <option timestep="1e-4"/>
    <option integrator="RK4"/>
    <option gravity="0 0 0"/>

    <asset>
        <texture type="skybox" builtin="gradient"
                 rgb1="1 1 1" rgb2="0.5 0.5 0.5"
                 width="265" height="256"/>
        <texture name="grid" type="2d" builtin="checker"
                 rgb1="0.1 0.1 0.1" rgb2="0.6 0.6 0.6"
                 width="300" height="300"/>
        <material name="grid" texture="grid"
                  texrepeat="10 10" reflectance="0.2"/>
    </asset>

    <worldbody>
        <light pos="0 0 10"/>

        <camera name="side view" pos="0.1 -1.5 1.0" euler="90 0 0" fovy="60"/>
        <camera name="upper view" pos="0 0 1.5" euler="0 0 0"/>

        <!-- fixed wall with tendon anchors -->
        <body name="wall" pos="0 0 0" euler="0 90 0">
            <geom type="plane" size="0.05 0.05 0.01" material="grid"/>
            <site name="t1_wall" pos="{R1 / 2} 0 0" type="sphere" size="0.002"/>
            <site name="t2_wall" pos="{-R1 / 2} 0 0" type="sphere" size="0.002"/>
        </body>

        <!-- intermediate bodies for tendons -->
        <body name="mid_body_t1" pos="{a + c / 2} 0 0">
            <site name="t1_mid" pos="0 0 0" type="sphere" size="0.001"/>
            <joint name="mid_joint_x_t1" type="slide" axis="1 0 0"/>
            <joint name="mid_joint_y_t1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size="0.002" mass="0.0001"
                  rgba="0.86 0.43 0.54 0.5" contype="0"/>
        </body>

        <body name="mid_body_t2" pos="{a + c / 2} 0 0">
            <site name="t2_mid" pos="0 0 0" type="sphere" size="0.001"/>
            <joint name="mid_joint_x_t2" type="slide" axis="1 0 0"/>
            <joint name="mid_joint_y_t2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size="0.002" mass="0.0001"
                  rgba="0.86 0.43 0.54 0.5" contype="0"/>
        </body>

        <!-- effector support body (still passive) -->
        <body name="effector_link" pos="{a + b + c} 0 0">
            <site name="effector_world" pos="0 0 0" type="sphere" size="0.001"/>
            <joint name="effector_x" type="slide" axis="1 0 0"/>
            <joint name="effector_y" type="slide" axis="0 0 1"/>
            <geom type="sphere" size="0.002" mass="0.0001"
                  rgba="0.86 0.43 0.54 0.5" contype="0"/>
        </body>

        <!-- kinematic chain: link1 -> link2 (pulley1) -> link3 (pulley2) -->
        <body name="link1" pos="0 0 0" euler="0 0 0">         
            <geom type="cylinder"
                  pos="{a / 2} 0 0"
                  size="0.002 {a / 2}"
                  euler="0 90 0"
                  rgba="0.21 0.32 0.82 0.5"
                  contype="0"/>

            <body name="link2" pos="{a} 0 0" euler="0 0 0">
                <joint name="A" type="hinge" axis="0 1 0"
                       stiffness="0" springref="0" damping="0"/>     
                <geom type="cylinder"
                      pos="{c / 2} 0 0"
                      size="0.002 {c / 2}"
                      euler="0 90 0"
                      rgba="0.42 0.32 0.12 0.5"
                      contype="0"/>

                <!-- pulley 1 -->
                <geom name="pulley1" type="cylinder"
                      size="{R1 / 2} 0.001"
                      pos="0 0 0"
                      euler="90 0 0"
                      rgba="0.42 0.32 0.12 0.5"
                      contype="0"/>
                <site name="side_r1_t1" pos="0 0 {-R1 / 2 - 0.002}"
                      type="sphere" size="0.001"/>
                <site name="side_r1_t2" pos="0 0 {R1 / 2 + 0.002}"
                      type="sphere" size="0.001"/>
                <site name="pulley1_side" pos="0 0 0"
                      type="sphere" size="0.001"/>

                <body name="link3" pos="{c} 0 0" euler="0 0 0">
                    <joint name="B" type="hinge" axis="0 1 0"
                           stiffness="0" springref="0" damping="0"/>     
                    <geom type="cylinder"
                          pos="{b / 2} 0 0"
                          size="0.002 {b / 2}"
                          euler="0 90 0"
                          rgba="0.34 0.65 0.84 0.5"
                          contype="0"/>
                    <geom type="box"
                          pos="{b} 0 0"
                          size="0.002 0.002 {R2 / 2}"
                          rgba="0.34 0.65 0.84 0.5"
                          mass="0"
                          contype="0"/>

                    <site name="t1_end" pos="{b} 0 {R2 / 2}"
                          type="sphere" size="0.002"/>
                    <site name="t2_end" pos="{b} 0 {-R2 / 2}"
                          type="sphere" size="0.002"/>

                    <!-- pulley 2 -->
                    <geom name="pulley2" type="cylinder"
                          size="{R2 / 2} 0.001"
                          pos="0 0 0"
                          euler="90 0 0"
                          rgba="0.34 0.65 0.84 0.5"
                          contype="0"/>
                    <site name="side_r2_t1" pos="0 0 {R2 / 2 + 0.002}"
                          type="sphere" size="0.001"/>
                    <site name="side_r2_t2" pos="0 0 {-R2 / 2 - 0.002}"
                          type="sphere" size="0.001"/>
                    <site name="pulley2_side" pos="0 0 0"
                          type="sphere" size="0.001"/>

                    <site name="effector" pos="{b} 0 0"
                          type="sphere" size="0.001"/>
                </body>
            </body>
        </body>
    </worldbody>

    <!-- tendons: passive, no actuators -->
    <tendon>
        <spatial name="tendon1_1" width="0.001"
                 stiffness="{stiffness}" damping="10"
                 springlength="0.005">
            <site site="t1_wall"/>
            <geom geom="pulley1" sidesite="side_r1_t1"/>
            <site site="t1_mid"/>
            <geom geom="pulley2" sidesite="side_r2_t1"/>
            <site site="t1_end"/>
        </spatial>
    </tendon>

    <tendon>
        <spatial name="tendon2_1" width="0.001"
                 stiffness="{stiffness}" damping="10"
                 springlength="0.005">
            <site site="t2_wall"/>
            <geom geom="pulley1" sidesite="side_r1_t2"/>
            <site site="t2_mid"/>
            <geom geom="pulley2" sidesite="side_r2_t2"/>
            <site site="t2_end"/>
        </spatial>
    </tendon>

    <!-- equality constraints to keep mid bodies between pulleys
         and fix effector orientation -->


</mujoco>
"""
