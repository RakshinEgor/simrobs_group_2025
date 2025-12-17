import numpy as np
import matplotlib.pyplot as plt

a = -4.71  # Represents Inertia (mass) of the robot joint
b = -0.27  # Represents Damping (friction) in the joint
c = 5.53   # Represents Stiffness (like a spring or controller gain)
d = -3.77  # Represents External Force (motor torque or gravity)

def robot_joint_dynamics(x):
    """
    Your specific ODE: a*x'' + b*x' + c*x = d
    State vector x = [y1, y2] = [position, velocity]
    """
    # x[0] is y1 (current position)
    # x[1] is y2 (current velocity)
    y1 = x[0]
    y2 = x[1]
    
    # Derivatives of the state vector:
    # y1_dot = y2
    # y2_dot = (d - b*y2 - c*y1) / a  (its for x_doubledot)
    
    y1_dot = y2
    y2_dot = (d - b * y2 - c * y1) / a
    
    return np.array([y1_dot, y2_dot])


def forward_euler(fun, x0, Tf, h):
    """
    Explicit Euler integration method
    (Fast, simple, but can be unstable)
    """
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t))) 
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k] + h * fun(x_hist[:, k])
    
    return x_hist, t

def backward_euler(fun, x0, Tf, h, tol=1e-8, max_iter=100):
    """
    Implicit Euler integration method (fixed-point iteration)
    (Slow, but very stable, good for 'stiff' problems like contact)
    """
    t = np.arange(0, Tf + h, h)
    x_hist = np.zeros((len(x0), len(t)))
    x_hist[:, 0] = x0
    
    for k in range(len(t) - 1):
        x_hist[:, k + 1] = x_hist[:, k]  # Initial guess
        
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
    (Industry standard: good balance of accuracy and speed)
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


#initial condition

initial_position = 1.0  # x(0) - Starting position of the joint
initial_velocity = 0.0  # x'(0) - Starting velocity of the joint
# ---------------------------------------------------

x0 = np.array([initial_position, initial_velocity])  # Initial state vector
Tf = 20.0  # Total simulation time (seconds)
h = 0.1    # Step size (seconds). we can change it with 0.1, 0.01, and 1

#
x_fe, t_fe = forward_euler(robot_joint_dynamics, x0, Tf, h)
x_be, t_be = backward_euler(robot_joint_dynamics, x0, Tf, h)
x_rk4, t_rk4 = runge_kutta4(robot_joint_dynamics, x0, Tf, h)

# --- (NEW) Calculate the Analytical (Exact) Solution ---
# These are the parameters we calculated earlier
r1 = -1.1124
r2 = 1.0551
xp = -0.6817

# ----- CHANGE ME: Calculate C1 and C2 based on your initial conditions -----
# (This calculation uses the placeholder initial_position=1.0, initial_velocity=0.0)
# 1. C1 + C2 = initial_position - xp  => C1 + C2 = 1.0 - (-0.6817) = 1.6817
# 2. r1*C1 + r2*C2 = initial_velocity => -1.1124*C1 + 1.0551*C2 = 0
#
# From (2), C2 = (1.1124 / 1.0551) * C1 = 1.0543 * C1
# Sub into (1): C1 + 1.0543*C1 = 1.6817 => 2.0543*C1 = 1.6817 => C1 = 0.8186
# C2 = 1.0543 * 0.8186 = 0.8631

C1 = 0.8186  
C2 = 0.8631  
# -------------------------------------------------------------------------

# Calculate the exact position at all time steps
x_analytical = C1 * np.exp(r1 * t_rk4) + C2 * np.exp(r2 * t_rk4) + xp

# --- (NEW) Plot the Results for Your Report ---
plt.figure(figsize=(24, 8))
plt.suptitle(f'Robot Joint Simulation: $a={a}, b={b}, c={c}, d={d}$ (h={h})', fontsize=16)

# Plot 1: Position vs. Time (The main comparison plot)
plt.subplot(1, 3, 1)
# x_fe[0, :] means: in the 'x_fe' array, get the 0th row (position) and all columns (time)
plt.plot(t_fe, x_fe[0, :], 'r--', label='Explicit Euler', alpha=0.8)
plt.plot(t_be, x_be[0, :], 'g:', label='Implicit Euler', alpha=0.8)
plt.plot(t_rk4, x_rk4[0, :], 'b.', label='RK4', markersize=6)
plt.plot(t_rk4, x_analytical, 'k-', label='Analytical Solution (Exact)', linewidth=2) 
plt.xlabel('Time (t)')
plt.ylabel('Position x(t)')
plt.legend()
plt.title('Position vs. Time Comparison')
plt.grid(True)

# Plot 2: Velocity vs. Time
plt.subplot(1, 3, 2)
# x_fe[1, :] means: get the 1st row (velocity)
plt.plot(t_fe, x_fe[1, :], 'r--', label='Explicit Euler', alpha=0.8)
plt.plot(t_be, x_be[1, :], 'g:', label='Implicit Euler', alpha=0.8)
plt.plot(t_rk4, x_rk4[1, :], 'b.', label='RK4', markersize=6)
plt.xlabel('Time (t)')
plt.ylabel('Velocity v(t)')
plt.legend()
plt.title('Velocity vs. Time')
plt.grid(True)

# Plot 3: Phase Portrait (Position vs. Velocity)
plt.subplot(1, 3, 3)
plt.plot(x_fe[0, :], x_fe[1, :], 'r--', label='Explicit Euler', alpha=0.8)
plt.plot(x_be[0, :], x_be[1, :], 'g:', label='Implicit Euler', alpha=0.8)
plt.plot(x_rk4[0, :], x_rk4[1, :], 'b', label='RK4')
# Calculate analytical velocity for the phase plot
v_analytical = C1*r1*np.exp(r1*t_rk4) + C2*r2*np.exp(r2*t_rk4)
plt.plot(x_analytical, v_analytical, 'k-', label='Analytical') 
plt.xlabel('Position x(t)')
plt.ylabel('Velocity v(t)')
plt.legend()
plt.title('Phase Portrait')
plt.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for the main title
plt.show()