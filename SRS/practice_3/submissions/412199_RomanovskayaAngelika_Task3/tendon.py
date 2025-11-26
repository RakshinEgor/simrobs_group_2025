import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import glfw

def simulate_with_controls():
    # XML модель
    xml = '''
<?xml version="1.0"?>
<mujoco>
    <option timestep="0.001"/>
    <option gravity="0 0 -9.8"/>
    
    <worldbody>
        <light pos="0 0 2"/>
        <geom name="ground" type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>
        <camera name="fixed" pos="0 -1.5 0.5" xyaxes="1 0 0 0 1 1"/>
        
        <body name="base_O1" pos="-0.5 0 0.1">
            <geom type="box" size="0.05 0.05 0.1" rgba="0.5 0.5 0.5 1"/>
            <site name="tendon1_start_top" type="sphere" size="0.005" pos="0.05 0 0.045" rgba="1 0.5 0 1"/>
            <site name="tendon2_start_bottom" type="sphere" size="0.005" pos="0.05 0 -0.045" rgba="0.5 0 1 1"/>
        </body>
        
        <body name="pulley1_assembly" pos="-0.3 0 0.1">
            <joint name="joint1" type="hinge" axis="0 0 1" range="-360 360"/>
            <geom type="cylinder" size="0.045 0.01" rgba="1 0 0 0.7" euler="90 0 0"/>
            <site name="pulley1_top" type="sphere" size="0.003" pos="0 0 0.045" rgba="1 0.5 0 1"/>
            <site name="pulley1_bottom" type="sphere" size="0.003" pos="0 0 -0.045" rgba="0.5 0 1 1"/>
        </body>
        
        <body name="pulley2_assembly" pos="0.1 0 0.1">
            <joint name="joint2" type="hinge" axis="0 0 1" range="-360 360"/>
            <geom type="cylinder" size="0.039 0.01" rgba="0 0 1 0.7" euler="90 0 0"/>
            <site name="pulley2_top" type="sphere" size="0.003" pos="0 0 0.039" rgba="1 0.5 0 1"/>
            <site name="pulley2_bottom" type="sphere" size="0.003" pos="0 0 -0.039" rgba="0.5 0 1 1"/>
        </body>
        
        <body name="base_O2" pos="0.3 0 0.1">
            <joint name="slider_O2" type="slide" axis="0 0 1" range="0 0.3" damping="0.5"/>
            <geom type="box" size="0.05 0.05 0.1" rgba="0.5 0.5 0.5 1" mass="2.0"/>
            <site name="tendon1_end_bottom" type="sphere" size="0.005" pos="-0.05 0 -0.039" rgba="1 0.5 0 1"/>
            <site name="tendon2_end_top" type="sphere" size="0.005" pos="-0.05 0 0.039" rgba="0.5 0 1 1"/>
        </body>
    </worldbody>

    <tendon>
        <spatial name="tendon1_cross" limited="true" range="0.5 2.0" width="0.003" rgba="1 0.5 0 0.8"
                 stiffness="800" damping="4">
            <site site="tendon1_start_top"/>
            <site site="pulley1_top"/>
            <site site="pulley2_bottom"/>
            <site site="tendon1_end_bottom"/>
        </spatial>
        
        <spatial name="tendon2_cross" limited="true" range="0.5 2.0" width="0.003" rgba="0.5 0 1 0.8"
                 stiffness="800" damping="4">
            <site site="tendon2_start_bottom"/>
            <site site="pulley1_bottom"/>
            <site site="pulley2_top"/>
            <site site="tendon2_end_top"/>
        </spatial>
    </tendon>

    <actuator>
        <position name="slider_motor" joint="slider_O2" gear="50" ctrllimited="true" ctrlrange="0 0.3"/>
    </actuator>
</mujoco>
'''

    # Загрузка модели
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Переменные для управления
    current_height = 0.1
    manual_control = False

    def key_callback(key, scancode, action, mods):
        nonlocal current_height, manual_control
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_UP:
                current_height = min(current_height + 0.01, 0.3)
                manual_control = True
            elif key == glfw.KEY_DOWN:
                current_height = max(current_height - 0.01, 0.0)
                manual_control = True
            elif key == glfw.KEY_SPACE:
                manual_control = False

    print("Tendon Connected 2R Planar Mechanism - Manual Control")
    print("=" * 55)
    print("Controls:")
    print("  UP ARROW    - Move load up")
    print("  DOWN ARROW  - Move load down") 
    print("  SPACE       - Switch to automatic mode")
    print("  ESC         - Exit")
    print("\nWatch how the crossed tendons transfer motion to the pulleys!")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            elapsed = time.time() - start_time
            
            if manual_control:
                # Ручное управление
                data.ctrl[0] = current_height
            else:
                # Автоматическое синусоидальное движение
                data.ctrl[0] = 0.1 + 0.1 * math.sin(2 * math.pi * 0.3 * elapsed)
                current_height = data.ctrl[0]
            
            # Шаг симуляции
            mujoco.mj_step(model, data)
            
            # Обновление viewer
            viewer.sync()
            
            time.sleep(model.opt.timestep)

    print("Simulation finished.")

if __name__ == "__main__":
    simulate_with_controls()