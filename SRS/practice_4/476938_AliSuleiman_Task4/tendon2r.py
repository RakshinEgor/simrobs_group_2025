import mujoco
import mujoco.viewer
import numpy as np
import math
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
from matplotlib import patches

def load_model():
    """Load the MuJoCo model from XML file"""
    xml_file = "tendon_2r_mechanism.xml"
    
    if not os.path.exists(xml_file):
        print(f"❌ Error: XML file '{xml_file}' not found!")
        print("   Please make sure the XML file is in the same directory as this script.")
        return None, None
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_file)
        data = mujoco.MjData(model)
        print("✅ Model loaded successfully!")
        print(f"   Number of actuators: {model.nu}")
        print(f"   Number of sensors: {model.nsensor}")
        return model, data
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def parse_xml_parameters(xml_file):
    """Parse parameters from XML file"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        print("✅ XML parameters parsed successfully!")
        return {}
    except Exception as e:
        print(f"❌ Error parsing XML: {e}")
        return None

def save_mechanism_screenshot(model, data, filename):
    """Save a screenshot of the mechanism using MuJoCo renderer"""
    try:
        # Create renderer and update scene
        renderer = mujoco.Renderer(model)
        renderer.update_scene(data)
        
        # Render and save image
        image = renderer.render()
        plt.imsave(filename, image)
        print(f"✅ Mechanism screenshot saved as {filename}")
    except Exception as e:
        print(f"❌ Error saving mechanism screenshot: {e}")

def plot_performance(time_history, q1_des_history, q2_des_history, 
                    q1_curr_history, q2_curr_history, control_history):
    """Generate performance plots for the report"""
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Convert to degrees for plotting
    q1_des_deg = np.degrees(q1_des_history)
    q2_des_deg = np.degrees(q2_des_history)
    q1_curr_deg = np.degrees(q1_curr_history)
    q2_curr_deg = np.degrees(q2_curr_history)
    
    # Figure 1: Tracking Performance
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_history, q1_des_deg, 'r--', label='Desired q1', linewidth=2)
    plt.plot(time_history, q1_curr_deg, 'r-', label='Actual q1', linewidth=1)
    plt.ylabel('Joint A Angle (deg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Joint Position Tracking Performance')
    
    plt.subplot(2, 1, 2)
    plt.plot(time_history, q2_des_deg, 'b--', label='Desired q2', linewidth=2)
    plt.plot(time_history, q2_curr_deg, 'b-', label='Actual q2', linewidth=1)
    plt.ylabel('Joint B Angle (deg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/tracking_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Control Effort
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, control_history[:, 0], 'r-', label='Motor A Torque', linewidth=1.5)
    plt.plot(time_history, control_history[:, 1], 'b-', label='Motor B Torque', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Torque (Nm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Control Effort')
    plt.savefig('images/control_effort.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Tracking Errors
    q1_error = q1_des_deg - q1_curr_deg
    q2_error = q2_des_deg - q2_curr_deg
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_history, q1_error, 'r-', label='Joint A Error', linewidth=1.5)
    plt.plot(time_history, q2_error, 'b-', label='Joint B Error', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error (deg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Joint Tracking Errors')
    
    # Add error statistics to plot
    max_error1 = np.max(np.abs(q1_error))
    max_error2 = np.max(np.abs(q2_error))
    rms_error1 = np.sqrt(np.mean(q1_error**2))
    rms_error2 = np.sqrt(np.mean(q2_error**2))
    
    plt.text(0.02, 0.98, f'Joint A: Max Error = {max_error1:.2f}°, RMS = {rms_error1:.2f}°', 
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.02, 0.90, f'Joint B: Max Error = {max_error2:.2f}°, RMS = {rms_error2:.2f}°', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.savefig('images/error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Performance plots generated successfully!")
    return max_error1, max_error2, rms_error1, rms_error2

def pd_control(model, data, time):
    """PD controller for joints A and B with sinusoidal reference signals"""
    
    # Convert degrees to radians for control parameters
    # q1 parameters (joint A)
    AMP1 = np.deg2rad(21.45)   # 21.45° in radians
    FREQ1 = 2.37               # 2.37 Hz
    BIAS1 = np.deg2rad(-0.4)   # -0.4° in radians
    
    # q2 parameters (joint B)  
    AMP2 = np.deg2rad(13.06)   # 13.06° in radians
    FREQ2 = 2.71               # 2.71 Hz
    BIAS2 = np.deg2rad(-9.7)   # -9.7° in radians
    
    # Desired positions (sinusoidal reference)
    q1_des = AMP1 * np.sin(2 * np.pi * FREQ1 * time) + BIAS1
    q2_des = AMP2 * np.sin(2 * np.pi * FREQ2 * time) + BIAS2
    
    # PD control gains (using the ones defined in XML)
    kp = 500  # proportional gain
    kv = 50   # derivative gain
    
    # Get current positions and velocities
    q1_curr = data.joint("A").qpos[0]
    q2_curr = data.joint("B").qpos[0]
    q1_vel = data.joint("A").qvel[0]
    q2_vel = data.joint("B").qvel[0]
    
    # PD control law
    tau1 = kp * (q1_des - q1_curr) + kv * (0 - q1_vel)
    tau2 = kp * (q2_des - q2_curr) + kv * (0 - q2_vel)
    
    # Apply control signals
    data.ctrl[0] = tau1  # motor_A
    data.ctrl[1] = tau2  # motor_B
    
    return q1_des, q2_des, q1_curr, q2_curr, tau1, tau2

def simulate_with_control(model, data, duration=10.0):
    """Run simulation with PD control and generate figures"""
    
    print(f"\n🎮 Starting simulation with PD control for {duration} seconds")
    print("=" * 60)
    print("Control parameters:")
    print(f"  Joint A (q1): AMP={21.45}°, FREQ={2.37}Hz, BIAS={-0.4}°")
    print(f"  Joint B (q2): AMP={13.06}°, FREQ={2.71}Hz, BIAS={-9.7}°")
    print(f"  PD gains: kp=500, kv=50")
    
    # Initialize lists to store data for plotting
    time_history = []
    q1_des_history = []
    q2_des_history = []
    q1_curr_history = []
    q2_curr_history = []
    control_history = []
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    # Run simulation with viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set optimal camera view
        viewer.cam.distance = 0.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        
        # Reset simulation time
        data.time = 0
        
        # Take initial screenshot
        save_mechanism_screenshot(model, data, 'images/mechanism_diagram.png')
        
        # Main simulation loop
        while viewer.is_running() and data.time < duration:
            # Apply PD control
            q1_des, q2_des, q1_curr, q2_curr, tau1, tau2 = pd_control(model, data, data.time)
            
            # Store data for analysis
            time_history.append(data.time)
            q1_des_history.append(q1_des)
            q2_des_history.append(q2_des)
            q1_curr_history.append(q1_curr)
            q2_curr_history.append(q2_curr)
            control_history.append([tau1, tau2])
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer with current state
            viewer.sync()
            
            # Print debug information occasionally
            if int(data.time * 10) % 50 == 0 and data.time > 0:  # Print every 5 seconds
                print(f"Time: {data.time:.1f}s | "
                      f"q1: {np.rad2deg(q1_curr):.1f}° (des: {np.rad2deg(q1_des):.1f}°) | "
                      f"q2: {np.rad2deg(q2_curr):.1f}° (des: {np.rad2deg(q2_des):.1f}°)")
        
        # Take final screenshot
        save_mechanism_screenshot(model, data, 'images/mechanism_final_position.png')
    
    print("✅ Simulation completed!")
    
    # Convert to numpy arrays for analysis
    time_history = np.array(time_history)
    q1_des_history = np.array(q1_des_history)
    q2_des_history = np.array(q2_des_history)
    q1_curr_history = np.array(q1_curr_history)
    q2_curr_history = np.array(q2_curr_history)
    control_history = np.array(control_history)
    
    # Generate performance plots
    max_error1, max_error2, rms_error1, rms_error2 = plot_performance(
        time_history, q1_des_history, q2_des_history, 
        q1_curr_history, q2_curr_history, control_history
    )
    
    return (time_history, q1_des_history, q2_des_history, 
            q1_curr_history, q2_curr_history, control_history,
            max_error1, max_error2, rms_error1, rms_error2)

def main():
    """Main function to run the simulation with PD control and generate figures"""
    
    # Load model from XML file
    model, data = load_model()
    if model is None:
        return
    
    # Parse XML parameters
    xml_file = "tendon_2r_mechanism.xml"
    parameters = parse_xml_parameters(xml_file)
    if parameters is None:
        return
    
    # Run simulation with PD control and generate figures
    results = simulate_with_control(model, data, duration=10.0)
    
    # Unpack results
    (time_history, q1_des, q2_des, q1_curr, q2_curr, control_history,
     max_error1, max_error2, rms_error1, rms_error2) = results
    
    # Print performance summary
    print(f"\n📊 Performance Summary:")
    print(f"  Joint A - Max error: {max_error1:.2f}°, RMS error: {rms_error1:.2f}°")
    print(f"  Joint B - Max error: {max_error2:.2f}°, RMS error: {rms_error2:.2f}°")
    
    # Print where images are saved
    print(f"\n📁 Generated images saved in 'images/' directory:")
    print(f"  - mechanism_diagram.png: Mechanism overview")
    print(f"  - mechanism_final_position.png: Final configuration")
    print(f"  - tracking_performance.png: Joint tracking performance")
    print(f"  - control_effort.png: Control torque requirements")
    print(f"  - error_analysis.png: Tracking error analysis")

if __name__ == "__main__":
    main()