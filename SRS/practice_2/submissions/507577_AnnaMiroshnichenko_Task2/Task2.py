import numpy as np
import matplotlib.pyplot as plt

# Параметры системы
m = 0.7
k = 14
b = 0.03
x0 = 0.48
v0 = 0

# Аналитическое решение для сравнения
def analytical_solution(t):
    alpha = b / (2 * m)
    omega_0 = np.sqrt(k / m)
    omega_d = np.sqrt(omega_0**2 - alpha**2)
    C1 = x0
    C2 = (v0 + alpha * x0) / omega_d
    x = np.exp(-alpha * t) * (C1 * np.cos(omega_d * t) + C2 * np.sin(omega_d * t))
    v = np.exp(-alpha * t) * ((-alpha * C1 + omega_d * C2) * np.cos(omega_d * t) - 
                             (alpha * C2 + omega_d * C1) * np.sin(omega_d * t))
    return x, v

# Явный метод Эйлера
def explicit_euler(dt, T):
    n = int(T / dt)
    t = np.zeros(n)
    x = np.zeros(n)
    v = np.zeros(n)
    
    x[0] = x0
    v[0] = v0
    
    for i in range(n-1):
        t[i+1] = t[i] + dt
        a = -(b * v[i] + k * x[i]) / m
        x[i+1] = x[i] + v[i] * dt
        v[i+1] = v[i] + a * dt
    
    return t, x, v

# Неявный метод Эйлера
def implicit_euler(dt, T):
    n = int(T / dt)
    t = np.zeros(n)
    x = np.zeros(n)
    v = np.zeros(n)
    
    x[0] = x0
    v[0] = v0
    
    denom = m + b*dt + k*dt**2
    
    for i in range(n-1):
        t[i+1] = t[i] + dt
        x[i+1] = (m*x[i] + m*v[i]*dt + b*x[i]*dt) / denom
        v[i+1] = (m*v[i] - k*x[i]*dt) / (m + b*dt)
    
    return t, x, v

# Метод Рунге-Кутты 4-го порядка
def runge_kutta_4(dt, T):
    n = int(T / dt)
    t = np.zeros(n)
    x = np.zeros(n)
    v = np.zeros(n)
    
    x[0] = x0
    v[0] = v0
    
    def derivatives(x_val, v_val):
        dxdt = v_val
        dvdt = -(b * v_val + k * x_val) / m
        return dxdt, dvdt
    
    for i in range(n-1):
        t[i+1] = t[i] + dt
        
        k1x, k1v = derivatives(x[i], v[i])
        k2x, k2v = derivatives(x[i] + 0.5*dt*k1x, v[i] + 0.5*dt*k1v)
        k3x, k3v = derivatives(x[i] + 0.5*dt*k2x, v[i] + 0.5*dt*k2v)
        k4x, k4v = derivatives(x[i] + dt*k3x, v[i] + dt*k3v)
        
        x[i+1] = x[i] + (dt/6) * (k1x + 2*k2x + 2*k3x + k4x)
        v[i+1] = v[i] + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
    
    return t, x, v

# Параметры интегрирования
T = 20  # общее время
dt = 0.1  # шаг времени

# Вычисление решений
t_analytical = np.linspace(0, T, 1000)
x_analytical, v_analytical = analytical_solution(t_analytical)

t_euler, x_euler, v_euler = explicit_euler(dt, T)
t_implicit, x_implicit, v_implicit = implicit_euler(dt, T)
t_rk4, x_rk4, v_rk4 = runge_kutta_4(dt, T)

# Построение графиков
plt.figure(figsize=(14, 12))

# Основной график
plt.subplot(2, 2, 1)
plt.plot(t_analytical, x_analytical, 'k-', linewidth=2, label='Аналитическое решение', alpha=0.8)
plt.plot(t_euler, x_euler, 'r--', linewidth=1.5, label=f'Явный Эйлер (dt={dt})', marker='o', markersize=3, markevery=5)
plt.plot(t_implicit, x_implicit, 'g--', linewidth=1.5, label=f'Неявный Эйлер (dt={dt})', marker='s', markersize=3, markevery=5)
plt.plot(t_rk4, x_rk4, 'b--', linewidth=1.5, label=f'Рунге-Кутта 4 (dt={dt})', marker='^', markersize=3, markevery=5)

plt.grid(True, alpha=0.3)
plt.xlabel('Время t, с')
plt.ylabel('Смещение x(t), м')
plt.title('Сравнение численных методов решения ОДУ\n$m\\ddot{x} + b\\dot{x} + kx = 0$')
plt.legend()
plt.ylim(-0.6, 0.6)

# График ошибок
plt.subplot(2, 2, 2)
x_analytical_euler = analytical_solution(t_euler)[0]
x_analytical_implicit = analytical_solution(t_implicit)[0]
x_analytical_rk4 = analytical_solution(t_rk4)[0]

error_euler = np.abs(x_euler - x_analytical_euler)
error_implicit = np.abs(x_implicit - x_analytical_implicit)
error_rk4 = np.abs(x_rk4 - x_analytical_rk4)

plt.plot(t_euler, error_euler, 'r-', linewidth=1, label='Ошибка явного Эйлера', alpha=0.7)
plt.plot(t_implicit, error_implicit, 'g-', linewidth=1, label='Ошибка неявного Эйлера', alpha=0.7)
plt.plot(t_rk4, error_rk4, 'b-', linewidth=1, label='Ошибка Рунге-Кутты 4', alpha=0.7)

plt.grid(True, alpha=0.3)
plt.xlabel('Время t, с')
plt.ylabel('Абсолютная ошибка, м')
plt.title('Ошибки численных методов')
plt.legend()
plt.yscale('log')

# ФАЗОВЫЙ ПОРТРЕТ
plt.subplot(2, 2, 3)
plt.plot(x_analytical, v_analytical, 'k-', linewidth=2, label='Аналитическое решение', alpha=0.8)
plt.plot(x_euler, v_euler, 'r--', linewidth=1.5, label='Явный Эйлер', marker='o', markersize=3, markevery=10)
plt.plot(x_implicit, v_implicit, 'g--', linewidth=1.5, label='Неявный Эйлер', marker='s', markersize=3, markevery=10)
plt.plot(x_rk4, v_rk4, 'b--', linewidth=1.5, label='Рунге-Кутта 4', marker='^', markersize=3, markevery=10)

# Начальная точка
plt.plot(x0, v0, 'ko', markersize=8, label='Начальные условия')

plt.grid(True, alpha=0.3)
plt.xlabel('Смещение x, м')
plt.ylabel('Скорость v, м/с')
plt.title('Фазовый портрет системы')
plt.legend()
plt.axis('equal')

plt.tight_layout()
plt.show()

# Сравнение точности методов
print("Сравнение методов (средняя абсолютная ошибка):")
print(f"Явный Эйлер: {np.mean(error_euler):.6f} м")
print(f"Неявный Эйлер: {np.mean(error_implicit):.6f} м")
print(f"Рунге-Кутта 4: {np.mean(error_rk4):.6f} м")
print(f"\nМаксимальная ошибка:")
print(f"Явный Эйлер: {np.max(error_euler):.6f} м")
print(f"Неявный Эйлер: {np.max(error_implicit):.6f} м")
print(f"Рунге-Кутта 4: {np.max(error_rk4):.6f} м")
