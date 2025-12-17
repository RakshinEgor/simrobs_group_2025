import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple


# Базовый класс системы
class System:
    def derive(self, state: np.ndarray) -> np.ndarray:
        pass


# Класс системы маятника
@dataclass
class PendulumSystem(System):
    m_kg: float
    k_Nm: float
    b_Nsm: float
    l_m: float
    theta_0_rad: float
    x_0_m: float
    g_ms2: float = 9.81

    def derive(self, state: np.ndarray) -> np.ndarray:
        x, dx = state
        # Уравнение: ml²θ'' + bθ' + kθ + mgl·sinθ = 0
        # θ'' = -(bθ' + kθ + mgl·sinθ)/(ml²)
        ddx = -((self.b_Nsm * dx + self.k_Nm * x +
                 self.m_kg * self.g_ms2 * self.l_m * np.sin(x)) /
                (self.m_kg * self.l_m * self.l_m))
        return np.array([dx, ddx])


# Базовый класс решателя
class Solver:
    def __init__(self, sys: System):
        self.sys = sys

    def solve(self, x0: np.ndarray, Tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        pass


# Метод явного Эйлера
class ForwardEuler(Solver):
    def solve(self, x0: np.ndarray, Tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(0, Tf, h)
        x_hist = np.zeros((len(x0), len(t)))
        x_hist[:, 0] = x0

        for k in range(len(t) - 1):
            x_hist[:, k + 1] = x_hist[:, k] + h * self.sys.derive(x_hist[:, k])

        return x_hist, t


# Метод неявного Эйлера
class BackwardEuler(Solver):
    def solve(self, x0: np.ndarray, Tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(0, Tf + h, h)
        x_hist = np.zeros((len(x0), len(t)))
        x_hist[:, 0] = x0

        for k in range(len(t) - 1):
            e = 1  # начальное значение ошибки
            x_hist[:, k + 1] = x_hist[:, k]  # начальное приближение

            while e > 1e-10:
                x_n = x_hist[:, k] + h * self.sys.derive(x_hist[:, k + 1])
                e = np.linalg.norm(x_n - x_hist[:, k + 1])  # вычисление ошибки
                x_hist[:, k + 1] = x_n

        return x_hist, t


# Метод Рунге-Кутты 4 порядка
class RungeKutta(Solver):
    def f_rk4(self, xk: np.ndarray, h: float) -> np.ndarray:
        f1 = self.sys.derive(xk)
        f2 = self.sys.derive(xk + 0.5 * h * f1)
        f3 = self.sys.derive(xk + 0.5 * h * f2)
        f4 = self.sys.derive(xk + h * f3)
        return xk + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    def solve(self, x0: np.ndarray, Tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(0, Tf + h, h)
        x_hist = np.zeros((len(x0), len(t)))
        x_hist[:, 0] = x0

        for k in range(len(t) - 1):
            x_hist[:, k + 1] = self.f_rk4(x_hist[:, k], h)

        return x_hist, t


# Класс для аналитического решения (линеаризованная система)
class AnalyticalSolver:
    def __init__(self, sys: PendulumSystem):
        self.sys = sys

    def solve(self, t: np.ndarray) -> np.ndarray:
        """
        Аналитическое решение для линеаризованной системы
        Уравнение: ml²θ'' + bθ' + (k + mgl)θ = 0
        """
        m = self.sys.m_kg
        l = self.sys.l_m
        b = self.sys.b_Nsm
        k = self.sys.k_Nm
        g = self.sys.g_ms2
        theta0 = self.sys.theta_0_rad

        # Параметры линеаризованного уравнения
        A = m * l ** 2
        B = b
        C = k + m * g * l  # линеаризация: sinθ ≈ θ

        # Характеристическое уравнение: Aλ² + Bλ + C = 0
        discriminant = B ** 2 - 4 * A * C

        theta = np.zeros_like(t)
        dtheta = np.zeros_like(t)

        if discriminant >= 0:
            # Перезатухающий или критически затухающий случай
            lambda1 = (-B + np.sqrt(discriminant)) / (2 * A)
            lambda2 = (-B - np.sqrt(discriminant)) / (2 * A)

            if abs(lambda1 - lambda2) < 1e-10:
                # Критическое затухание
                C1 = theta0
                C2 = -lambda1 * theta0
                theta = (C1 + C2 * t) * np.exp(lambda1 * t)
                dtheta = (C2 + lambda1 * (C1 + C2 * t)) * np.exp(lambda1 * t)
            else:
                # Перезатухание
                C2 = theta0 * lambda1 / (lambda1 - lambda2)
                C1 = theta0 - C2
                theta = C1 * np.exp(lambda1 * t) + C2 * np.exp(lambda2 * t)
                dtheta = C1 * lambda1 * np.exp(lambda1 * t) + C2 * lambda2 * np.exp(lambda2 * t)
        else:
            # Затухающие колебания
            alpha = -B / (2 * A)
            beta = np.sqrt(-discriminant) / (2 * A)

            C1 = theta0
            C2 = (alpha * theta0) / beta  # начальная скорость = 0

            theta = np.exp(alpha * t) * (C1 * np.cos(beta * t) + C2 * np.sin(beta * t))
            dtheta = alpha * np.exp(alpha * t) * (C1 * np.cos(beta * t) + C2 * np.sin(beta * t)) + \
                     np.exp(alpha * t) * (-C1 * beta * np.sin(beta * t) + C2 * beta * np.cos(beta * t))

        x_hist = np.vstack([theta, dtheta])
        return x_hist


# Функция для сравнительного анализа всех шагов
def analyze_different_steps():
    """Анализ методов при разных шагах интегрирования с раздельными выводами"""

    sys = PendulumSystem(
        m_kg=0.8,
        k_Nm=7,
        b_Nsm=0.035,
        l_m=0.61,
        theta_0_rad=-1.036757776,
        x_0_m=0.98,
        g_ms2=9.81
    )

    x_0 = [sys.theta_0_rad, 0]
    Tf = 5

    time_steps = [0.001, 0.01, 0.1]

    print("=" * 60)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ ИНТЕГРИРОВАНИЯ")
    print("=" * 60)

    for Ts in time_steps:
        print(f"\n АНАЛИЗ ДЛЯ ШАГА Ts = {Ts}")
        print("-" * 40)

        # Моделирование
        x_hist_fwd, t_fwd = ForwardEuler(sys).solve(x_0, Tf, Ts)
        x_hist_bkwd, t_bkwd = BackwardEuler(sys).solve(x_0, Tf, Ts)
        x_hist_rk4, t_rk4 = RungeKutta(sys).solve(x_0, Tf, Ts)

        # Аналитическое решение
        t_common = np.arange(0, Tf, Ts)
        analytical_solver = AnalyticalSolver(sys)
        x_hist_analytical = analytical_solver.solve(t_common)

        # Обрезаем до общего размера
        min_len = min(len(t_fwd), len(t_bkwd), len(t_rk4), len(t_common))
        t_common = t_common[:min_len]
        x_hist_analytical = x_hist_analytical[:, :min_len]

        # Вычисляем ошибки
        error_fwd = np.abs(x_hist_fwd[0, :min_len] - x_hist_analytical[0, :])
        error_bkwd = np.abs(x_hist_bkwd[0, :min_len] - x_hist_analytical[0, :])
        error_rk4 = np.abs(x_hist_rk4[0, :min_len] - x_hist_analytical[0, :])

        # Среднеквадратичные ошибки
        rmse_fwd = np.sqrt(np.mean(error_fwd ** 2))
        rmse_bkwd = np.sqrt(np.mean(error_bkwd ** 2))
        rmse_rk4 = np.sqrt(np.mean(error_rk4 ** 2))

        # Максимальные ошибки
        max_error_fwd = np.max(error_fwd)
        max_error_bkwd = np.max(error_bkwd)
        max_error_rk4 = np.max(error_rk4)

        print("Среднеквадратичные ошибки (RMSE):")
        print(f"  • Явный Эйлер:    {rmse_fwd:.6f} rad")
        print(f"  • Неявный Эйлер:  {rmse_bkwd:.6f} rad")
        print(f"  • Рунге-Кутта 4:  {rmse_rk4:.6f} rad")

        print("\nМаксимальные ошибки:")
        print(f"  • Явный Эйлер:    {max_error_fwd:.6f} rad")
        print(f"  • Неявный Эйлер:  {max_error_bkwd:.6f} rad")
        print(f"  • Рунге-Кутта 4:  {max_error_rk4:.6f} rad")


        # Визуализация для текущего шага
        plt.figure(figsize=(12, 8))

        # График решений
        plt.subplot(2, 1, 1)
        plt.plot(t_common, x_hist_analytical[0, :], 'k-', label='Analytical', linewidth=2)
        plt.plot(t_fwd[:min_len], x_hist_fwd[0, :min_len], 'r--', label='Forward Euler', alpha=0.8)
        plt.plot(t_bkwd[:min_len], x_hist_bkwd[0, :min_len], 'b--', label='Backward Euler', alpha=0.8)
        plt.plot(t_rk4[:min_len], x_hist_rk4[0, :min_len], 'g--', label='RK4', alpha=0.8)
        plt.ylabel('$\\theta$ [rad]')
        plt.title(f'Сравнение методов интегрирования (Ts = {Ts})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # График ошибок
        plt.subplot(2, 1, 2)
        plt.plot(t_common, error_fwd, 'r-', label='Forward Euler', alpha=0.8)
        plt.plot(t_common, error_bkwd, 'b-', label='Backward Euler', alpha=0.8)
        plt.plot(t_common, error_rk4, 'g-', label='RK4', alpha=0.8)
        plt.ylabel('Absolute Error [rad]')
        plt.xlabel('Time [sec]')
        plt.title('Ошибки методов интегрирования')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyze_different_steps()