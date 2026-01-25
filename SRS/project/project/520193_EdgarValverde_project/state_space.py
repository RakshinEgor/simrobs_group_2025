import numpy as np
from math import factorial

def get_matrices(deep, dgrs, dt):
    """
    deep: Order of the system (e.g., 4 for Snap control: Pos, Vel, Acc, Jerk)
    dgrs: Degrees of freedom (joints)
    dt: Time step
    """
    Ai = np.eye(deep)
    for i in range(deep):
        for j in range(i + 1, deep):
            Ai[i, j] = (dt**(j - i)) / factorial(j - i)
            
    # B matrix: Impact of input u on derivatives
    # If deep=4 (Snap control), u affects Snap directly, then integrates down.
    # States: [Pos, Vel, Acc, Jerk]. 
    # Input u is Snap.
    Bi = np.zeros((deep, 1))
    for i in range(deep):
        # The power of dt corresponds to how many integrals away it is
        # Position (idx 0) is result of 4 integrals -> dt^4 / 4! (if deep=4)
        # Actually, if x_dot = Ax + Bu:
        # Index deep-1 (Jerk) gets dt^1/1! * u
        # Index 0 (Pos) gets dt^deep/deep! * u
        power = deep - i 
        Bi[i, 0] = (dt**power) / factorial(power)

    # 2. Block Diagonal Expansion for all joints
    A = np.zeros((deep * dgrs, deep * dgrs))
    B = np.zeros((deep * dgrs, dgrs))

    for i in range(dgrs):
        row = i * deep
        col = i * deep
        A[row:row+deep, col:col+deep] = Ai
        B[row:row+deep, i] = Bi.flatten()
        
    return A, B