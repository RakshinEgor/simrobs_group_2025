import time

import mujoco
import mujoco.viewer

paused = False


def key_callback(keycode):
    global paused
    if keycode == 32:  # Space key
        paused = not paused


def main():
    global paused

    # Загружаем обновленную модель
    m = mujoco.MjModel.from_xml_path('optimusknee.xml')
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        start = time.time()

        # Устанавливаем начальное положение для анимации
        d.qpos[0] = 0.5  # Начальный угол для кривошипа

        while viewer.is_running():
            step_start = time.time()

            if not paused:


                mujoco.mj_step(m, d)

                # Синхронизация визуализации
                viewer.sync()

                # Регулировка скорости для реального времени
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()