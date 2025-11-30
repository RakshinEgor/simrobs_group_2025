import numpy as np
import matplotlib.pyplot as plt

def pendulum_system(state):
    theta, dtheta = state
    m = 0.8
    b = 0.01
    l = 0.83
    g = 9.8
    ddtheta = - (b/(m*l**2)) * dtheta - (g/l) * np.sin(theta)
    return np.array([dtheta, ddtheta])

def pendulum_bckw_euler(fun, x0, t_f, h):
    t = np.arange(0, t_f + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0

    for k in range(len(t) - 1):
        e = 1
        x_hist[:, k + 1] = x_hist[:, k]
        while e > 1e-8:
            x_n = x_hist[:, k] + h * fun(x_hist[:, k + 1])
            e = np.linalg.norm(x_n - x_hist[:, k + 1])
            x_hist[:, k + 1] = x_n

    return x_hist, t

theta_0 = -0.1570845115
dtheta_0 = 0
x0 = np.array([theta_0, dtheta_0])

x_hist, t_hist = pendulum_bckw_euler(pendulum_system, x0, 10, 0.01)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_hist, x_hist[0, :], 'b-', label="$\\theta$")
plt.xlabel('Время, [сек]')
plt.ylabel('Угол, [рад]')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_hist, x_hist[1, :], 'r-', label="$d\\theta/dt$")
plt.xlabel('Время, [сек]')
plt.ylabel('Угловая скорость, [рад/сек]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x_hist[0, :], x_hist[1, :], 'g-')
plt.xlabel('Угол $\\theta$, [рад]')
plt.ylabel('Угловая скорость $d\\theta/dt$, [рад/сек]')
plt.grid(True)
plt.show()
