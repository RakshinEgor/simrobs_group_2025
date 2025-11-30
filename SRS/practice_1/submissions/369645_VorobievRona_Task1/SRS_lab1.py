
import numpy as np
import matplotlib.pyplot as plt

def forward_euler(fun, x0, Tf, h):
    """
    Explicit Euler integration method
    """
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k] + h * fun(t[k], x_hist[:, k])
    
    return x_hist, t

def backward_euler(fun, x0, Tf, h):
    """
    Implicit Euler integration method using fixed-point iteration
    """
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k] + h*fun(t[k] + 0.5*h, x_hist[:, k] + 0.5*h*fun(t[k], x_hist[:, k]))  # Initial guess
        
    return x_hist, t

def runge_kutta4(fun, x0, Tf, h):
    """
    4th order Runge-Kutta integration method
    """
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        k1 = fun(t[k], x_hist[:, k])
        k2 = fun(t[k] + 0.5*h, x_hist[:, k] + 0.5 * h * k1)
        k3 = fun(t[k] + 0.5*h, x_hist[:, k] + 0.5 * h * k2)
        k4 = fun(t[k] + h, x_hist[:, k] + h * k3)
        
        x_hist[:, k + 1] = x_hist[:, k] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return x_hist, t

#Analytic solution
def analitic_solution(Tf, h):
    t = np.arange(0, Tf + h, h)
    D = b**2 - 4*a*c
    p1 = 0.5*(-b - np.sqrt(D))/a
    p2 = 0.5*(-b + np.sqrt(D))/a
    return 1.135*np.exp(p1*t) + 1.233*np.exp(p2*t) - 1.368, t


a = 8.57
b = -6.64
c = -4.86
d = 6.65

#Solution parameters
Tf = 10.0    
h = 0.01      
y0 = 1.0          
z0 = 1.0         
    
#Function for system of DEs
# y' = z
# z' = (d - b*z - c*y)/a

def f(x, y_vec):
    y, z = y_vec
    return np.array([z, (d - b*z - c*y)/a])
    
    
x0 = np.array([y0, z0])

x_an, t_an = analitic_solution(Tf, h)
    
# Forward Eiler
x_fe, t_fe = forward_euler(f, x0, Tf, h)
    
# Backward Eiler
x_be, t_be = backward_euler(f, x0, Tf, h)
    
# Runge-Kutta
x_rk4, t_rk4 = runge_kutta4(f, x0, Tf, h)

fe_err = np.abs(x_fe[0, :] - x_an)
be_err = np.abs(x_be[0, :] - x_an)
rk4_err = np.abs(x_rk4[0, :] - x_an)

print("Результаты в конечной точке:")
print(f"Аналитическое решение: x({10}) = {x_an[-1]:.6f}")
print(f"Явный Эйлер:     x({10}) = {x_fe[0, -1]:.6f}, погрешность: {fe_err[-1]:.0f}")
print(f"Неявный Эйлер:   x({10}) = {x_be[0, -1]:.6f}, погрешность: {be_err[-1]:.0f}")
print(f"Рунге-Кутта 4:   x({10}) = {x_rk4[0, -1]:.6f}, погрешность: {rk4_err[-1]:.0f}")

# Plot results
plt.figure(figsize=(24, 8))

plt.plot(t_an, x_an, label=f'Analytic')
plt.plot(t_fe, x_fe[0, :], label=f'Forward Euler: {fe_err[-1]:.0f}')
plt.plot(t_be, x_be[0, :], label=f'Backward Euler: {be_err[-1]:.0f}')
plt.plot(t_rk4, x_rk4[0, :], label=f'RK4: {rk4_err[-1]:.0f}')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()

plt.tight_layout()
plt.show()
