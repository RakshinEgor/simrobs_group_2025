import mujoco
import mujoco.viewer
import numpy as np
import math
import xml.etree.ElementTree as ET
import os

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
        return model, data
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None




def show_dimensions_in_viewer(model, data, parameters):
    """Display the mechanism with dimension visualization"""
    
    print("\n📐 Displaying Mechanism Dimensions in MuJoCo Viewer")
    print("=" * 50)
    print("   The viewer will show the mechanism with its key dimensions.")
    print("   Close the viewer window to exit.")
    
    # Run viewer without simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set optimal camera view
        viewer.cam.distance = 0.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        
    
        
        # Keep the viewer open until user closes it
        print(f"\n🖥️  Viewer is running... Close the window to exit.")
        
        while viewer.is_running():
            # No simulation steps - just keep the viewer alive
            pass
    
    #print("\n✅ Dimension display completed!")


def main():
    """Main function to display dimensions"""
    
    
    
    # Load model from XML file
    model, data = load_model()
    if model is None:
        return
    
    # Parse XML parameters
    xml_file = "tendon_2r_mechanism.xml"
    parameters = parse_xml_parameters(xml_file)
    if parameters is None:
        return
    
   
    
   
    
    # Show dimensions in viewer
    show_dimensions_in_viewer(model, data, parameters)

if __name__ == "__main__":
    main()