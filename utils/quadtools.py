import numpy as np


def angle_wrap(angle):
    """
    Wrap angle to [-pi, pi)
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_yaw(quat):
    """
    Extract yaw angle from quaternion [x, y, z, w]
    Return yaw in radians.
    """
    x, y, z, w = quat
    
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    
    return np.arctan2(siny_cosp, cosy_cosp)


def quat_to_euler(quat):
    """
    Convert quaternion [x, y, z, w] to Euler angles (roll, pitch, yaw)
    All angles are in radians.

    Returns:
        roll  (rotation around x-axis)
        pitch (rotation around y-axis)
        yaw   (rotation around z-axis)
    """
    x, y, z, w = quat
    
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw
