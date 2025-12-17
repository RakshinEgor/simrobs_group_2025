import mujoco
import mujoco.viewer
import numpy as np
import os
import time

R1 = 0.012  # Радиус первого шкива
R2 = 0.03  # Радиус второго шкива
a = 0.062  # Расстояние от левой стенки до центра block1
b = 0.048  # Расстояние от центра block2 до нагрузки
c = 0.056  # Расстояние между центрами шкивов


def crossed_tangent_points(c1, r1, c2, r2):  # Возвращает точки касания двух перекрёстных касательны

    c1, c2 = np.asarray(c1), np.asarray(c2)
    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)
    if d < 1e-8:
        return None

    # Единичный вектор от c1 к c2 и перпендикуляр
    u = d_vec / d
    v = np.array([-u[1], u[0]])  # поворот на 90° против часовой

    # Эффективный радиус второго шкива для внутренних касательных — отрицательный
    r2_eff = -r2
    denom = r1 - r2_eff  # = r1 + r2

    # Косинус угла между линией центров и направлением на точку касания
    cos_alpha = denom / d
    if abs(cos_alpha) > 1.0:
        return None
    sin_alpha = np.sqrt(1 - cos_alpha ** 2)

    points = []
    for sign in [1, -1]:
        direction = cos_alpha * u + sign * sin_alpha * v
        p1 = c1 + r1 * direction
        p2 = c2 + r2_eff * direction
        points.append((p1, p2))
    return points


def update_tendon_sites(model, data):
    block1_pos = data.body("body_block1").xpos[:2]
    block2_pos = data.body("body_block2").xpos[:2]

    tangents = crossed_tangent_points(block1_pos, R1, block2_pos, R2)
    if tangents is None:
        return

    # Сортируем по Y на первом шкиве: t1 — верхняя, t2 — нижняя
    tangents = sorted(tangents, key=lambda t: t[0][1], reverse=True)
    (t1_b1, t1_b2), (t2_b1, t2_b2) = tangents

    z = 0.0

    # Преобразуем глобальные координаты в локальные относительно тела
    def global_to_body(pos_global, body_name):
        xpos = data.body(body_name).xpos
        xmat = data.body(body_name).xmat.reshape(3, 3)
        vec = np.array([pos_global[0], pos_global[1], z]) - xpos
        return xmat.T @ vec

    # block1
    data.qpos[model.joint("t1_b1_x").qposadr[0]:model.joint("t1_b1_x").qposadr[0] + 2] = global_to_body(t1_b1,
                                                                                                        "body_block1")[
                                                                                         :2]
    data.qpos[model.joint("t2_b1_x").qposadr[0]:model.joint("t2_b1_x").qposadr[0] + 2] = global_to_body(t2_b1,
                                                                                                        "body_block1")[
                                                                                         :2]

    # block2
    data.qpos[model.joint("t1_b2_x").qposadr[0]:model.joint("t1_b2_x").qposadr[0] + 2] = global_to_body(t1_b2,
                                                                                                        "body_block2")[
                                                                                         :2]
    data.qpos[model.joint("t2_b2_x").qposadr[0]:model.joint("t2_b2_x").qposadr[0] + 2] = global_to_body(t2_b2,
                                                                                                        "body_block2")[
                                                                                         :2]


def control_callback(model, data):
    # Синусоидальная нагрузка (можно менять амплитуду и частоту)
    amplitude = 0.06
    period = 3.0
    data.ctrl[0] = amplitude * np.sin(2 * np.pi * data.time / period)

    # Обновляем точки касания перед каждым шагом физики
    update_tendon_sites(model, data)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir_path, "model.xml")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    mujoco.set_mjcb_control(control_callback)

    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.001)
