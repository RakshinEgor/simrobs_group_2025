import numpy as np
import matplotlib.pyplot as plt

def system_dynamics(x):
    """
    Система: -9.42*ẍ + 4.79*ẋ - 0.06*x = -3.06
    Преобразуем к: ẍ = (4.79*ẋ - 0.06*x + 3.06)/9.42
    Состояние: x = [x, ẋ]
    """
    a = -9.42
    b = 4.79
    c = -0.06
    d = -3.06
    
    # ẍ = (b*ẋ + c*x + d)/(-a)
    x_dot = x[1]
    x_ddot = (b*x[1] + c*x[0] + d)/(-a)
    
    return np.array([x_dot, x_ddot])

def analytical_solution(t, x0):
    """
    Аналитическое решение: x(t) = C₁e^(0.4961t) + C₂e^(0.0129t) + 51
    """
    # Находим константы из начальных условий
    # x(0) = x0[0], ẋ(0) = x0[1]
    A = np.array([[1, 1], [0.4961, 0.0129]])
    b_vec = np.array([x0[0] - 51, x0[1]])
    C = np.linalg.solve(A, b_vec)
    
    return C[0]*np.exp(0.4961*t) + C[1]*np.exp(0.0129*t) + 51

def forward_euler(fun, x0, Tf, h):
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k] + h * fun(x_hist[:, k])
    
    return x_hist, t

def backward_euler(fun, x0, Tf, h, tol=1e-8, max_iter=100):
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k]  # Начальное приближение
        
        for i in range(max_iter):
            x_next = x_hist[:, k] + h * fun(x_hist[:, k + 1])
            error = np.linalg.norm(x_next - x_hist[:, k + 1])
            x_hist[:, k + 1] = x_next
            
            if error < tol:
                break
    
    return x_hist, t

def runge_kutta4(fun, x0, Tf, h):
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

# Параметры решения
x0 = np.array([1.0, 0.0])  # Начальные условия: x(0)=1, ẋ(0)=0
Tf = 20.0
h = 0.01

# Численные решения
x_fe, t_fe = forward_euler(system_dynamics, x0, Tf, h)
x_be, t_be = backward_euler(system_dynamics, x0, Tf, h)
x_rk4, t_rk4 = runge_kutta4(system_dynamics, x0, Tf, h)

# Аналитическое решение
x_analytical = analytical_solution(t_fe, x0)

# Сравнение результатов
plt.figure(figsize=(15, 10))

# Положение
plt.subplot(2, 2, 1)
plt.plot(t_fe, x_fe[0, :], label='Forward Euler', alpha=0.7)
plt.plot(t_be, x_be[0, :], label='Backward Euler', alpha=0.7)
plt.plot(t_rk4, x_rk4[0, :], label='RK4', alpha=0.7)
plt.plot(t_fe, x_analytical, 'k--', label='Analytical', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Position x(t)')
plt.legend()
plt.title('Position vs Time')
plt.grid(True)

# Скорость
plt.subplot(2, 2, 2)
plt.plot(t_fe, x_fe[1, :], label='Forward Euler', alpha=0.7)
plt.plot(t_be, x_be[1, :], label='Backward Euler', alpha=0.7)
plt.plot(t_rk4, x_rk4[1, :], label='RK4', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Velocity dx/dt')
plt.legend()
plt.title('Velocity vs Time')
plt.grid(True)

# Фазовый портрет
plt.subplot(2, 2, 3)
plt.plot(x_fe[0, :], x_fe[1, :], label='Forward Euler', alpha=0.7)
plt.plot(x_be[0, :], x_be[1, :], label='Backward Euler', alpha=0.7)
plt.plot(x_rk4[0, :], x_rk4[1, :], label='RK4', alpha=0.7)
plt.xlabel('Position x')
plt.ylabel('Velocity dx/dt')
plt.legend()
plt.title('Phase Portrait')
plt.grid(True)

# Ошибки
plt.subplot(2, 2, 4)
error_fe = np.abs(x_fe[0, :] - x_analytical)
error_be = np.abs(x_be[0, :] - x_analytical)
error_rk4 = np.abs(x_rk4[0, :] - x_analytical)

plt.semilogy(t_fe, error_fe, label='Forward Euler Error')
plt.semilogy(t_be, error_be, label='Backward Euler Error')
plt.semilogy(t_rk4, error_rk4, label='RK4 Error')
plt.xlabel('Time')
plt.ylabel('Absolute Error')
plt.legend()
plt.title('Numerical Errors (log scale)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Вывод максимальных ошибок
print("Maximum absolute errors:")
print(f"Forward Euler:{np.max(error_fe):.6e}")
print(f"Backward Euler: {np.max(error_be):.6e}")
print(f"Runge-Kutta 4: {np.max(error_rk4):.6e}")
<img width="1489" height="990" alt="загруженное" src="https://github.com/user-attachments/assets/3e03937e-2429-4646-8c3d-72f2a0d9d439" />
