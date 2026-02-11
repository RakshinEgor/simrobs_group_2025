import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# Параметры желаемой траектории из задания
AMP_DEG = 19.3
FREQ = 3.41
BIAS_DEG = 5.3

# Конвертация в радианы
AMP_RAD = np.deg2rad(AMP_DEG)
BIAS_RAD = np.deg2rad(BIAS_DEG)


KP = 100
KD = 5

XML_PATH = "4r_new.xml"

paused = False


def key_callback(keycode):
    global paused
    if keycode == 32:  # Space key
        paused = not paused


def get_sensor_value(model, data, sensor_name):
    """Получить значение сенсора по имени"""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid == -1:
        raise ValueError(f"Сенсор '{sensor_name}' не найден")
    return data.sensordata[model.sensor_adr[sid]]


def main():
    global paused

    # Загружаем модель
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Проверяем сенсор
    sensor_name = "sensor_O_pos"
    try:
        test_value = get_sensor_value(model, data, sensor_name)
        print(f"Сенсор '{sensor_name}' обнаружен. Начальное значение: {test_value}")
    except Exception as e:
        print(f"Ошибка сенсора: {e}")
        return

    # Инициализация
    data.qpos[0] = BIAS_RAD  # Начальное положение = смещение
    data.qvel[0] = 0.0

    # Логирование данных
    log_time = []
    log_q_des = []
    log_q_sensed = []
    log_error = []
    log_ctrl = []

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = np.array([0.05, 0.0, 0.02])
        viewer.cam.distance = 0.3
        viewer.cam.elevation = -20

        start_time = time.time()
        step = 0
        prev_sensed_pos = BIAS_RAD

        while viewer.is_running():
            step_start = time.time()
            current_time = time.time() - start_time

            # Останавливаем через 30 секунд
            if current_time >= 30.0:
                break

            if not paused:
                # Желаемая траектория
                q_des = AMP_RAD * np.sin(FREQ * current_time) + BIAS_RAD

                # Текущее положение с сенсора
                sensed_pos = get_sensor_value(model, data, sensor_name)

                # Численное дифференцирование для скорости
                if step == 0:
                    sensed_vel = 0.0
                else:
                    dt = model.opt.timestep
                    sensed_vel = (sensed_pos - prev_sensed_pos) / dt
                    prev_sensed_pos = sensed_pos

                # ПД-регулятор
                pos_error = q_des - sensed_pos
                u = KP * pos_error + KD * (-sensed_vel)  # Желаемая скорость = 0

                # Применяем управление
                data.ctrl[0] = u

                # Шаг симуляции
                mujoco.mj_step(model, data)

                # Логирование каждые 10 шагов
                if step % 10 == 0:
                    log_time.append(current_time)
                    log_q_des.append(np.rad2deg(q_des))
                    log_q_sensed.append(np.rad2deg(sensed_pos))
                    log_error.append(np.rad2deg(pos_error))
                    log_ctrl.append(u)

                step += 1

            # Синхронизация визуализации
            viewer.sync()

            # Регулировка скорости для реального времени
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Построение графиков
    if len(log_time) > 0:
        plt.figure(figsize=(12, 10))

        # График 1: Желаемое и фактическое положение
        plt.subplot(3, 1, 1)
        plt.plot(log_time, log_q_des, 'g--', label='$q^{des}(t)$', linewidth=1.5)
        plt.plot(log_time, log_q_sensed, 'b', label='$q_{sensed}(t)$', linewidth=1.2)
        plt.ylabel('Угол, °')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f'Переходный процесс (данные с сенсора)\n$K_p = {KP}$, $K_d = {KD}$')

        # График 2: Ошибка
        plt.subplot(3, 1, 2)
        plt.plot(log_time, log_error, 'r', label='Ошибка $e(t)$', linewidth=1.2)
        plt.axhline(0, color='k', linewidth=0.5, linestyle=':')
        plt.ylabel('Ошибка, °')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # График 3: Управляющее воздействие
        plt.subplot(3, 1, 3)
        plt.plot(log_time, log_ctrl, 'm', label='Управление $u(t)$', linewidth=1.2)
        plt.xlabel('Время, с')
        plt.ylabel('Момент, Н·м')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        plt.show()
    else:
        print("Нет данных для построения графика!")


if __name__ == "__main__":
    main()