import time
import mujoco
import mujoco.viewer

# --- Глобальный флаг для остановки симуляции ---
stop_simulation = False


def keyboard_callback(keycode):
    global stop_simulation

    # Преобразуем код в символ
    try:
        key = chr(keycode).lower()
    except:
        return

    if key == 'q':
        stop_simulation = True


# ---- Загрузка модели ----
model = mujoco.MjModel.from_xml_path(
    "SRS\\practice_3\\submissions\\336835_MischenkoIvan_Task3\\task3_model.xml")
data = mujoco.MjData(model)

# ---- Запуск viewer с обработчиком клавиш ----
with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_callback) as viewer:
    last_time = time.time()
    sim_dt = model.opt.timestep
    while viewer.is_running() and not stop_simulation:
        mujoco.mj_step(model, data)
        now = time.time()
        elapsed = now - last_time
        if elapsed < sim_dt:
            time.sleep(sim_dt - elapsed)
        last_time = now
        viewer.sync()
