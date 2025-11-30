import numpy as np
import matplotlib.pyplot as plt
a = -0.68
b = 3.91
c = 7.85
d = -0.8

def your_equation(x):
    """
    State vector x = [x, dx/dt]
    """
    x_pos = x[0]    # x
    x_vel = x[1]    # dx/dt
    
    # x'' = (d - b·x' - c·x) / a
    x_acc = (d - b*x_vel - c*x_pos) / a
    
    return np.array([x_vel, x_acc])

def analytic_solution(t):
    return 0.0181 * np.exp(7.326*t) + 0.0839 * np.exp(-1.576*t) - 0.102

def analytic_derivative(t):
    return 0.0181 * 7.326 * np.exp(7.326*t) + 0.0839 * (-1.576) * np.exp(-1.576*t)

def forward_euler(fun, x0, Tf, h):
    """
    Explicit Euler integration method
    """
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k] + h * fun(x_hist[:, k])
    
    return x_hist, t

def backward_euler(fun, x0, Tf, h, tol=1e-8, max_iter=100):
    """
    Implicit Euler integration method using fixed-point iteration
    """
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k] 
        
        for i in range(max_iter):
            x_next = x_hist[:, k] + h * fun(x_hist[:, k + 1])
            error = np.linalg.norm(x_next - x_hist[:, k + 1])
            x_hist[:, k + 1] = x_next
            
            if error < tol:
                break
    
    return x_hist, t

def runge_kutta4(fun, x0, Tf, h):
    """
    4th order Runge-Kutta integration method
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


x0 = np.array([0.0, 0.0])  # Начальные условия: x(0)=0, x'(0)=0
Tf = 1.0 
h = 0.01

x_fe, t_fe = forward_euler(your_equation, x0, Tf, h)
x_be, t_be = backward_euler(your_equation, x0, Tf, h)
x_rk4, t_rk4 = runge_kutta4(your_equation, x0, Tf, h)

plt.figure(figsize=(24, 8))


plt.subplot(1, 3, 1)
plt.plot(t_fe, x_fe[0, :], label='Forward Euler')
plt.plot(t_be, x_be[0, :], label='Backward Euler')
plt.plot(t_rk4, x_rk4[0, :], label='RK4')
plt.plot(t_fe, analytic_solution(t_fe), 'k--', label='Analytical', linewidth=3)  
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.legend()
plt.title('Solution x(t) vs Time')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(t_fe, x_fe[1, :], label='Forward Euler')
plt.plot(t_be, x_be[1, :], label='Backward Euler') 
plt.plot(t_rk4, x_rk4[1, :], label='RK4')
plt.plot(t_fe, analytic_derivative(t_fe), 'k--', label='Analytical Deriv', linewidth=3)
plt.xlabel('Time')
plt.ylabel('dx/dt')
plt.legend()
plt.title('Derivative dx/dt vs Time')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x_fe[0, :], x_fe[1, :], label='Forward Euler')
plt.plot(x_be[0, :], x_be[1, :], label='Backward Euler')
plt.plot(x_rk4[0, :], x_rk4[1, :], label='RK4')
plt.plot(analytic_solution(t_fe), analytic_derivative(t_fe), 'k--', 
         label='Analytical Phase', linewidth=3)
plt.xlabel('x(t)')
plt.ylabel('dx/dt')
plt.legend()
plt.title('Phase Portrait')
plt.grid(True)

plt.tight_layout()
plt.show()
