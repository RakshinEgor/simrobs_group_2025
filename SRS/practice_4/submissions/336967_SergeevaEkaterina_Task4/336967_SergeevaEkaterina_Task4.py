import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

XML_PATH = "optimus.xml"

# Параметры желаемой траектории
AMP_DEG = 17.31
FREQ = 3.72
BIAS_DEG = -43.9
AMP_RAD = np.deg2rad(AMP_DEG)
BIAS_RAD = np.deg2rad(BIAS_DEG)

# ПД-регулятор
KP = 50
KD = 0.001

def get_sensor_value(model, data, sensor_name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid == -1:
        raise ValueError(f"Сенсор '{sensor_name}' не найден")
    return data.sensordata[model.sensor_adr[sid]]

def main():
    print("Загрузка модели...")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    sensor_name = "sensor_0_pos"
    try:
        get_sensor_value(model, data, sensor_name)
        print(f"Сенсор '{sensor_name}' обнаружен.")
    except Exception as e:
        print(f"Ошибка сенсора: {e}")
        return

    # Инициализация
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    # Логгеры
    log_time = []
    log_q_des = []
    log_q_sensed = []
    log_ctrl = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = np.array([0.09, 0.0, 0.0])
        viewer.cam.distance = 0.5
        viewer.cam.elevation = -30

        t0 = time.time()
        step = 0
        prev_sensed_pos = 0.0

        while viewer.is_running():
            t = time.time() - t0

            if t >= 30.0:
                break

            q_des = AMP_RAD * np.sin(FREQ * t) + BIAS_RAD

            sensed_pos = get_sensor_value(model, data, sensor_name)

            if step == 0:
                sensed_vel = 0.0
            else:
                dt = model.opt.timestep
                sensed_vel = (sensed_pos - prev_sensed_pos) / dt
            prev_sensed_pos = sensed_pos

            # ПД-управление
            u = KP * (q_des - sensed_pos) - KD * sensed_vel
            data.ctrl[0] = u

            # Шаг симуляции
            mujoco.mj_step(model, data)
            viewer.sync()

            if step % 10 == 0:
                log_time.append(t)
                log_q_des.append(np.rad2deg(q_des))
                log_q_sensed.append(np.rad2deg(sensed_pos))
                log_ctrl.append(u)

            step += 1

    n_points = len(log_time)

    if n_points == 0:
        print("Нет данных для построения графика!!!")
        return

    log_time = np.array(log_time)
    log_q_des = np.array(log_q_des)
    log_q_sensed = np.array(log_q_sensed)
    log_error = log_q_des - log_q_sensed

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    # Исправленные строки - убраны лишние обратные слеши
    plt.plot(log_time, log_q_des, 'g--', label=r'$q^{\mathrm{des}}(t)$', linewidth=1.5)
    plt.plot(log_time, log_q_sensed, 'b', label=r'$q_{\mathrm{sensed}}(t)$', linewidth=1.2)
    plt.ylabel('Угол, °')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(log_time, log_error, 'r', label='Ошибка $e(t)$', linewidth=1.2)
    plt.axhline(0, color='k', linewidth=0.5, linestyle=':')
    plt.ylabel('Ошибка, °')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(log_time, log_ctrl, 'm', label='Управление $u(t)$', linewidth=1.2)
    plt.xlabel('Время, с')
    plt.ylabel('Момент, Н·м')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Подставляем KP и KD в заголовок
    plt.suptitle(f'Переходный процесс (данные с сенсора)\n$K_p = {KP}$, $K_d = {KD}$',
                fontsize=14, y=0.95)

    plt.tight_layout()


    plt.show()

if __name__ == "__main__":
    main()