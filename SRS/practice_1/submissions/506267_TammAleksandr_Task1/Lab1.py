import numpy as np
import matplotlib.pyplot as plt

# Коэффициенты из задания
a = -8.57
b = 7.85
c = -8.03
d = -2.16


def linear_ode(t, x):
    """
    ОДУ: a·ẍ + b·ẋ + c·x = d
    В форме: ẋ₁ = x₂, ẋ₂ = (d - c·x₁ - b·x₂)/a
    """
    return np.array([x[1], (d - c * x[0] - b * x[1]) / a])


def analytical_solution(t, A=0.731, B=-0.392):
    """Аналитическое решение"""
    alpha = 0.458
    beta = 0.853
    x_particular = 0.269

    x_pos = np.exp(alpha * t) * (A * np.cos(beta * t) + B * np.sin(beta * t)) + x_particular
    x_vel = np.exp(alpha * t) * ((alpha * A + beta * B) * np.cos(beta * t) + (alpha * B - beta * A) * np.sin(beta * t))

    return x_pos, x_vel


def explicit_euler(func, x0, t_span, dt):
    """Явный метод Эйлера"""
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    x = np.zeros((len(x0), len(t)))
    x[:, 0] = x0

    for i in range(len(t) - 1):
        x[:, i + 1] = x[:, i] + dt * func(t[i], x[:, i])

    return t, x


def implicit_euler(func, x0, t_span, dt, tol=1e-8, max_iter=100):
    """Неявный метод Эйлера"""
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    x = np.zeros((len(x0), len(t)))
    x[:, 0] = x0

    for i in range(len(t) - 1):
        # Начальное приближение
        x_guess = x[:, i]

        # Итерации Ньютона
        for _ in range(max_iter):
            f_next = func(t[i + 1], x_guess)
            x_new = x[:, i] + dt * f_next

            if np.linalg.norm(x_new - x_guess) < tol:
                break
            x_guess = x_new

        x[:, i + 1] = x_new

    return t, x


def runge_kutta4(func, x0, t_span, dt):
    """Метод Рунге-Кутты 4-го порядка"""
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    x = np.zeros((len(x0), len(t)))
    x[:, 0] = x0

    for i in range(len(t) - 1):
        k1 = func(t[i], x[:, i])
        k2 = func(t[i] + dt / 2, x[:, i] + dt / 2 * k1)
        k3 = func(t[i] + dt / 2, x[:, i] + dt / 2 * k2)
        k4 = func(t[i] + dt, x[:, i] + dt * k3)

        x[:, i + 1] = x[:, i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, x


# Параметры решения
x0 = np.array([1.0, 0.0])  # Начальные условия
t_span = [0, 5]  # Время моделирования
dt = 0.001  # Шаг

# Численные решения
t_fe, x_fe = explicit_euler(linear_ode, x0, t_span, dt)
t_be, x_be = implicit_euler(linear_ode, x0, t_span, dt)
t_rk, x_rk = runge_kutta4(linear_ode, x0, t_span, dt)

# Аналитическое решение
x_analytical, v_analytical = analytical_solution(t_fe)


# Анализ ошибок - ТОЛЬКО МАКСИМАЛЬНАЯ ОШИБКА
def calculate_max_error(analytical, numerical):
    """Вычисление только максимальной ошибки"""
    max_err = np.max(np.abs(analytical - numerical))
    return max_err


# Вычисляем только максимальные ошибки
max_errors = {
    'Explicit Euler': calculate_max_error(x_analytical, x_fe[0]),
    'Implicit Euler': calculate_max_error(x_analytical, x_be[0]),
    'Runge-Kutta 4': calculate_max_error(x_analytical, x_rk[0])
}

# Визуализация
plt.figure(figsize=(15, 10))

# Положение
plt.subplot(2, 2, 1)
plt.plot(t_fe, x_analytical, 'k-', linewidth=2, label='Analytical')
plt.plot(t_fe, x_fe[0], 'r--', label='Explicit Euler')
plt.plot(t_be, x_be[0], 'g--', label='Implicit Euler')
plt.plot(t_rk, x_rk[0], 'b--', label='Runge-Kutta 4')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Position vs Time')
plt.grid(True)

# Ошибки
plt.subplot(2, 2, 2)
plt.plot(t_fe, np.abs(x_analytical - x_fe[0]), 'r-', label='Explicit Euler')
plt.plot(t_be, np.abs(x_analytical - x_be[0]), 'g-', label='Implicit Euler')
plt.plot(t_rk, np.abs(x_analytical - x_rk[0]), 'b-', label='Runge-Kutta 4')
plt.xlabel('Time')
plt.ylabel('Absolute Error')
plt.legend()
plt.title('Absolute Errors')
plt.yscale('log')
plt.grid(True)

# Фазовый портрет
plt.subplot(2, 2, 3)
plt.plot(x_analytical, v_analytical, 'k-', label='Analytical')
plt.plot(x_fe[0], x_fe[1], 'r--', label='Explicit Euler')
plt.plot(x_be[0], x_be[1], 'g--', label='Implicit Euler')
plt.plot(x_rk[0], x_rk[1], 'b--', label='Runge-Kutta 4')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.legend()
plt.title('Phase Portrait')
plt.grid(True)

# Сравнение МАКСИМАЛЬНЫХ ошибок
plt.subplot(2, 2, 4)
methods = list(max_errors.keys())
max_err_values = [max_errors[m] for m in methods]

x = np.arange(len(methods))

plt.bar(x, max_err_values, width=0.6, color=['red', 'green', 'blue'], alpha=0.7)
plt.xlabel('Methods')
plt.ylabel('Max Error')
plt.title('Maximum Error Comparison')
plt.xticks(x, methods, rotation=45)
plt.grid(True, alpha=0.3)

# Добавляем значения на столбцы
for i, v in enumerate(max_err_values):
    plt.text(i, v + 0.001, f'{v:.6f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Вывод результатов
print("MAXIMUM ERROR ANALYSIS RESULTS")
print("=" * 50)
print(f"{'Method':<15} {'Max Error':<12}")
print("-" * 50)
for method, max_err in max_errors.items():
    print(f"{method:<15} {max_err:<12.6f}")

# Дополнительный анализ: ошибки скорости
print("\nMAXIMUM VELOCITY ERRORS")
print("=" * 50)
print(f"{'Method':<15} {'Max Vel Error':<12}")
print("-" * 50)
vel_errors = {
    'Explicit Euler': calculate_max_error(v_analytical, x_fe[1]),
    'Implicit Euler': calculate_max_error(v_analytical, x_be[1]),
    'Runge-Kutta 4': calculate_max_error(v_analytical, x_rk[1])
}

for method, max_err in vel_errors.items():
    print(f"{method:<15} {max_err:<12.6f}")