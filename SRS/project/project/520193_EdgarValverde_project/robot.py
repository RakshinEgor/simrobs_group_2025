import numpy as np

class Robot:
    def __init__(self, dh_params):
        """
        Initializes the robot with Denavit-Hartenberg parameters.
        dh_params: Dictionary {joint_index: {'theta':, 'd':, 'a':, 'alpha':}}
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)
        
    def _dh_matrix(self, theta, d, a, alpha):
        """Standard DH Transformation Matrix."""
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_alp, s_alp = np.cos(alpha), np.sin(alpha)
        
        # Standard DH Matrix
        return np.array([
            [c_th, -s_th*c_alp,  s_th*s_alp, a*c_th],
            [s_th,  c_th*c_alp, -c_th*s_alp, a*s_th],
            [0,     s_alp,       c_alp,      d],
            [0,     0,           0,          1]
        ])

    def get_transforms(self, q):
        """
        Calculates the transformation matrix for each link relative to base.
        Returns a list of 4x4 matrices [T0, T1, T2, ... T_end].
        """
        T_accum = []
        T_curr = np.eye(4)
        T_accum.append(T_curr.copy())

        for i in range(self.num_joints):
            params = self.dh_params[i+1]
            # Apply joint variable q[i] to theta offset
            theta_val = q[i] + params['theta']
            
            T_link = self._dh_matrix(theta_val, params['d'], params['a'], params['alpha'])
            T_curr = T_curr @ T_link
            T_accum.append(T_curr.copy())
            
        return T_accum

    def forward_kinematics(self, q):
        """Returns the End-Effector Transformation Matrix."""
        transforms = self.get_transforms(q)
        return transforms[-1]
    
    def get_stick_model(self, q):
        """
        Returns the (x, y, z) coordinates of each joint origin.
        Used for 3D visualization (wireframe/stick model).
        """
        transforms = self.get_transforms(q)
        # Extract the translation vector (column 3, rows 0-2) from each T matrix
        coords = np.array([T[:3, 3] for T in transforms])
        return coords

# --- Rotation Helpers ---
def euler_to_rot_matrix(roll, pitch, yaw):
    """Calculates Rotation Matrix from RPY angles (X-Y-Z convention)."""
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx