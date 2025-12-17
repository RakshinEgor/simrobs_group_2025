import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp

# Параметры ОДУ
a = 7.43
b = -3.47
c = -8.85
d = -7.63

def ode_system(x):
    """
    Система ОДУ второго порядка: a*x'' + b*x' + c*x = d
    Преобразование в систему первого порядка:
    x1 = x, x2 = x'
    x1' = x2
    x2' = (d - b*x2 - c*x1)/a
    """
    x1, x2 = x
    dx1dt = x2
    dx2dt = (d - b*x2 - c*x1) / a
    return np.array([dx1dt, dx2dt])

def forward_euler(fun, x0, Tf, h):
    """
    Явный метод Эйлера
    """
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k] + h * fun(x_hist[:, k])
    
    return x_hist, t

def backward_euler(fun, x0, Tf, h, tol=1e-8, max_iter=100):
    """
    Неявный метод Эйлера с итерационным решением
    """
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        # Начальное приближение - значение на предыдущем шаге
        x_next = x_hist[:, k]
        
        # Итерационный процесс (простой итерационный метод)
        for i in range(max_iter):
            x_next_new = x_hist[:, k] + h * fun(x_next)
            error = np.linalg.norm(x_next_new - x_next)
            x_next = x_next_new
            
            if error < tol:
                break
        
        x_hist[:, k + 1] = x_next
    
    return x_hist, t

def runge_kutta4(fun, x0, Tf, h):
    """
    Метод Рунге-Кутты 4-го порядка
    """
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        k1 = fun(x_hist[:, k])
        k2 = fun(x_hist[:, k] + 0.5 * h * k1)
        k3 = fun(x_hist[:, k] + 0.5 * h * k2)
        k4 = fun(x_hist[:, k] + h * k3)
        
        x_hist[:, k + 1] = x_hist[:, k] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return x_hist, t

def analytical_solution(t, x0):
    """
    Аналитическое решение ОДУ: a*x'' + b*x' + c*x = d
    Для начальных условий x(0) = x0[0], x'(0) = x0[1]
    """
    # Характеристическое уравнение
    D = b**2 - 4*a*c
    r1 = (-b + sqrt(D)) / (2*a)
    r2 = (-b - sqrt(D)) / (2*a)
    
    # Частное решение
    x_p = d / c
    
    # Определение констант из начальных условий
    # x(0) = C1 + C2 + x_p = x0[0]
    # x'(0) = r1*C1 + r2*C2 = x0[1]
    
    A = np.array([[1, 1], [r1, r2]])
    b_vec = np.array([x0[0] - x_p, x0[1]])
    C1, C2 = np.linalg.solve(A, b_vec)
    
    # Аналитическое решение
    x_analytical = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t) + x_p
    x_derivative = C1 * r1 * np.exp(r1 * t) + C2 * r2 * np.exp(r2 * t)
    
    return x_analytical, x_derivative, r1, r2, x_p, C1, C2

# Параметры интегрирования
x0 = np.array([1.0, 0.0])  # Начальные условия: x(0) = 1, x'(0) = 0
Tf = 5.0
h = 0.01

# Численное интегрирование
print("Численное интегрирование...")
x_fe, t_fe = forward_euler(ode_system, x0, Tf, h)
x_be, t_be = backward_euler(ode_system, x0, Tf, h)
x_rk4, t_rk4 = runge_kutta4(ode_system, x0, Tf, h)

# Аналитическое решение
print("Вычисление аналитического решения...")
x_analytical, x_analytical_deriv, r1, r2, x_p, C1, C2 = analytical_solution(t_fe, x0)

# Вывод параметров аналитического решения
print("\n" + "="*60)
print("АНАЛИТИЧЕСКОЕ РЕШЕНИЕ")
print("="*60)
print(f"Уравнение: {a}·x'' + {b}·x' + {c}·x = {d}")
print(f"Характеристические корни: r₁ = {r1:.4f}, r₂ = {r2:.4f}")
print(f"Частное решение: x_p = {x_p:.4f}")
print(f"Постоянные: C₁ = {C1:.4f}, C₂ = {C2:.4f}")
print(f"Общее решение: x(t) = {C1:.4f}·exp({r1:.4f}·t) + {C2:.4f}·exp({r2:.4f}·t) + {x_p:.4f}")

# Вычисление ошибок
error_fe = np.abs(x_fe[0, :] - x_analytical)
error_be = np.abs(x_be[0, :] - x_analytical)
error_rk4 = np.abs(x_rk4[0, :] - x_analytical)

print(f"\nСредние ошибки:")
print(f"Явный Эйлер: {np.mean(error_fe):.6f}")
print(f"Неявный Эйлер: {np.mean(error_be):.6f}")
print(f"Рунге-Кутта 4: {np.mean(error_rk4):.6f}")

# Построение графиков
plt.figure(figsize=(20, 12))

# График 1: Решения
plt.subplot(2, 3, 1)
plt.plot(t_fe, x_analytical, 'k-', linewidth=2, label='Аналитическое')
plt.plot(t_fe, x_fe[0, :], 'r--', label='Явный Эйлер')
plt.plot(t_be, x_be[0, :], 'g--', label='Неявный Эйлер')
plt.plot(t_rk4, x_rk4[0, :], 'b--', label='Рунге-Кутта 4')
plt.xlabel('Время, t')
plt.ylabel('x(t)')
plt.title('Сравнение решений')
plt.legend()
plt.grid(True)

# График 2: Производные
plt.subplot(2, 3, 2)
plt.plot(t_fe, x_analytical_deriv, 'k-', linewidth=2, label='Аналитическое')
plt.plot(t_fe, x_fe[1, :], 'r--', label='Явный Эйлер')
plt.plot(t_be, x_be[1, :], 'g--', label='Неявный Эйлер')
plt.plot(t_rk4, x_rk4[1, :], 'b--', label='Рунге-Кутта 4')
plt.xlabel('Время, t')
plt.ylabel("x'(t)")
plt.title('Производные решений')
plt.legend()
plt.grid(True)

# График 3: Ошибки
plt.subplot(2, 3, 3)
plt.semilogy(t_fe, error_fe, 'r-', label='Явный Эйлер')
plt.semilogy(t_be, error_be, 'g-', label='Неявный Эйлер')
plt.semilogy(t_rk4, error_rk4, 'b-', label='Рунге-Кутта 4')
plt.xlabel('Время, t')
plt.ylabel('Абсолютная ошибка')
plt.title('Ошибки численных методов')
plt.legend()
plt.grid(True)

# График 4: Фазовый портрет
plt.subplot(2, 3, 4)
plt.plot(x_analytical, x_analytical_deriv, 'k-', linewidth=2, label='Аналитическое')
plt.plot(x_fe[0, :], x_fe[1, :], 'r--', label='Явный Эйлер')
plt.plot(x_be[0, :], x_be[1, :], 'g--', label='Неявный Эйлер')
plt.plot(x_rk4[0, :], x_rk4[1, :], 'b--', label='Рунге-Кутта 4')
plt.xlabel('x(t)')
plt.ylabel("x'(t)")
plt.title('Фазовый портрет')
plt.legend()
plt.grid(True)

# График 5: Детальное сравнение (первые 2 секунды)
plt.subplot(2, 3, 5)
t_limit = 2.0
mask = t_fe <= t_limit
plt.plot(t_fe[mask], x_analytical[mask], 'k-', linewidth=3, label='Аналитическое')
plt.plot(t_fe[mask], x_fe[0, mask], 'ro', markersize=3, label='Явный Эйлер')
plt.plot(t_be[mask], x_be[0, mask], 'g^', markersize=3, label='Неявный Эйлер')
plt.plot(t_rk4[mask], x_rk4[0, mask], 'bs', markersize=3, label='Рунге-Кутта 4')
plt.xlabel('Время, t')
plt.ylabel('x(t)')
plt.title('Детальное сравнение (первые 2 секунды)')
plt.legend()
plt.grid(True)

# График 6: Накопление ошибки во времени
plt.subplot(2, 3, 6)
plt.plot(t_fe, np.cumsum(error_fe), 'r-', label='Явный Эйлер')
plt.plot(t_be, np.cumsum(error_be), 'g-', label='Неявный Эйлер')
plt.plot(t_rk4, np.cumsum(error_rk4), 'b-', label='Рунге-Кутта 4')
plt.xlabel('Время, t')
plt.ylabel('Накопленная ошибка')
plt.title('Накопление ошибки во времени')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Анализ устойчивости
print("\n" + "="*60)
print("АНАЛИЗ УСТОЙЧИВОСТИ И ТОЧНОСТИ")
print("="*60)
print(f"Характеристические корни: r₁ = {r1:.4f}, r₂ = {r2:.4f}")

if r1 > 0 or r2 > 0:
    print("СИСТЕМА НЕУСТОЙЧИВА: имеется положительный корень")
    print("Ожидается экспоненциальный рост решения")
else:
    print("СИСТЕМА УСТОЙЧИВА: все корни отрицательные")

print(f"\nШАГ ИНТЕГРИРОВАНИЯ: h = {h}")
print("РЕКОМЕНДАЦИИ:")
print("1. Для неустойчивых систем предпочтительны неявные методы")
print("2. Метод Рунге-Кутты 4 обычно обеспечивает лучшую точность")
print("3. Явный Эйлер может быть неустойчив при больших шагах")
<img width="1990" height="1189" alt="image" src="https://github.com/user-attachments/assets/ea323da9-736b-4a1b-8a47-1e66b2e1623c" />
