import pandas as pd
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def play_trajectory(csv_filename="trajectory_data.csv"):
    
    try:
        df = pd.read_csv(csv_filename)
        print(f"📄 Datos cargados: {len(df)} pasos.")
    except FileNotFoundError:
        print("❌ Error: No se encuentra 'trajectory_data.csv'.")
        return

    client = RemoteAPIClient()
    sim = client.require('sim')
    print("🔌 Conectado a CoppeliaSim.")

    base_name = "/IRB140"
    joint_paths = [
        "./joint",                                          
        "./joint/link/joint",                               
        "./joint/link/joint/link/joint",                    
        "./joint/link/joint/link/joint/link/joint",         
        "./joint/link/joint/link/joint/link/joint/link/joint",             
        "./joint/link/joint/link/joint/link/joint/link/joint/link/joint"   
    ]
    joint_handles = []
    tip_handle = None

    try:
        base_handle = sim.getObject(base_name)
        for path in joint_paths:
            joint_handles.append(sim.getObject(path, {'proxy': base_handle}))
        
        try: tip_handle = sim.getObject("/IRB140/tip")
        except: tip_handle = sim.getObject("/IRB140/connection")
            
    except Exception as e:
        print(f"\n❌ Error buscando objetos: {e}")
        return

    if tip_handle is None:
        print("❌ No se encontró el tip.")
        return

    time.sleep(1.0) 
    client.setStepping(True) 
    sim.startSimulation()
    
    line_size = 4
    max_items = 99999
    color_trace = [1, 0, 1] 
    drawing_container = sim.addDrawingObject(sim.drawing_lines, line_size, 0, -1, max_items, color_trace)

    prev_pos = None

    try:
        for idx, row in df.iterrows():
            angles = [row[f'q{i+1}'] for i in range(6)]
            for i, angle in enumerate(angles):
                sim.setJointTargetPosition(joint_handles[i], angle)
            
            sim.step()
            
            current_pos = sim.getObjectPosition(tip_handle, -1)
            if prev_pos is not None:
                sim.addDrawingObjectItem(drawing_container, prev_pos + current_pos)
            prev_pos = current_pos
            
            if idx % 10 == 0:
                print(f"Paso {idx}/{len(df)}", end='\r')

        print(f"\n✅ Trayectoria finalizada.")
        
        time.sleep(2)
        sim.stopSimulation()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sim.stopSimulation()

if __name__ == "__main__":
    play_trajectory()