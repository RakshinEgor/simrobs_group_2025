import numpy as np
import matplotlib.pyplot as plt

m = 0.9
k = 9.2
b = 0.03

alpha = -0.0167
beta = 3.1972

C1 = 0.9
C2 = 0.0047

def pendulum_dynamics(x):

    x_dot_0 = x[0]
    x_dot_1 = x[1]
    x_dot_2 = (-b * x_dot_1 - k * x_dot_0)/m
    return np.array([x_dot_1, x_dot_2])

def analitical_solution(x0, Tf, h):

    t_val = np.arange(0, Tf + h, h)
    x_dot_0 = np.exp(alpha * t_val) * (C1 * np.cos(beta * t_val) + C2 * np.sin(beta * t_val))
    x_dot_1 = np.exp(alpha * t_val) * ((alpha * C1 + C2 * beta) * np.cos(beta * t_val) +(alpha * C2 - C1 * beta) * np.sin(beta * t_val))

    return np.vstack((x_dot_0, x_dot_1)), t_val


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
        x_prev = x_hist[:, k]
        x_next = x_prev.copy()
        for i in range(max_iter):
            x_temp = x_prev + h * fun(x_next)
            if np.linalg.norm(x_temp - x_next) < tol:
                break
            x_next = x_temp
        x_hist[:, k + 1] = x_next
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
        x_hist[:, k + 1] = x_hist[:, k] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return x_hist, t


x0 = np.array([0.9, 0.0])
Tf = 10.0
h = 0.01



# Analitical solution
x_an, t_an = analitical_solution(x0, Tf, h)

x_fe, t_fe = forward_euler(pendulum_dynamics, x0, Tf, h)
x_be, t_be = backward_euler(pendulum_dynamics, x0, Tf, h)
x_rk4, t_rk4 = runge_kutta4(pendulum_dynamics, x0, Tf, h)


# RMSE
rmse_fe_pos = np.sqrt(np.mean((x_an[0, :] - x_fe[0, :]) ** 2))
rmse_fe_vel = np.sqrt(np.mean((x_an[1, :] - x_fe[1, :]) ** 2))

rmse_be_pos = np.sqrt(np.mean((x_an[0, :] - x_be[0, :]) ** 2))
rmse_be_vel = np.sqrt(np.mean((x_an[1, :] - x_be[1, :]) ** 2))

rmse_rk4_pos = np.sqrt(np.mean((x_an[0, :] - x_rk4[0, :]) ** 2))
rmse_rk4_vel = np.sqrt(np.mean((x_an[1, :] - x_rk4[1, :]) ** 2))

print(f"RMSE Position:")
print(f"  Forward Euler: {rmse_fe_pos:.6f}")
print(f"  Backward Euler: {rmse_be_pos:.6f}")
print(f"  RK4: {rmse_rk4_pos:.6f}")

print(f"RMSE Velocity:")
print(f"  Forward Euler: {rmse_fe_vel:.6f}")
print(f"  Backward Euler: {rmse_be_vel:.6f}")
print(f"  RK4: {rmse_rk4_vel:.6f}")

plt.figure(figsize=(24, 8))

# Pos
plt.subplot(1, 3, 1)
plt.plot(t_fe, x_fe[0, :], label='Forward Euler')
plt.plot(t_be, x_be[0, :], label='Backward Euler')
plt.plot(t_rk4, x_rk4[0, :], label='RK4')
plt.plot(t_an, x_an[0, :], label='Analytical', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Pendulum Position vs Time')

# Vel
plt.subplot(1, 3, 2)
plt.plot(t_fe, x_fe[1, :], label='Forward Euler')
plt.plot(t_be, x_be[1, :], label='Backward Euler')
plt.plot(t_rk4, x_rk4[1, :], label='RK4')
plt.plot(t_an, x_an[1, :], label='Analytical', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.title('Velocity vs Time')

# Phase Portrait
plt.subplot(1, 3, 3)
plt.plot(x_fe[0, :], x_fe[1, :], label='Forward Euler')
plt.plot(x_be[0, :], x_be[1, :], label='Backward Euler')
plt.plot(x_rk4[0, :], x_rk4[1, :], label='RK4')
plt.plot(x_an[0, :], x_an[1, :], label='Analytical', linestyle='--')
plt.xlabel('Pos')
plt.ylabel('Velocity')
plt.legend()
plt.title('Phase Portrait')

plt.tight_layout()
plt.show()