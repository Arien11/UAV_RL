import numpy as np


class Poly5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """
            5-th order polynomial at each Axis
            p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5(物理描述比如 p = p0 + v0t + 1/2at^2)
            where,
                p(0)   = c0
                v(0)   = p'(0)  = c1
                a(0)   = p''(0) = 2*c2 = a0 → c2 = a0/2
                p(T)   = p(T)   = c0 + c1*T + c2*T^2 + c3*T^3 + c4*T^4 + c5*T^5
                v(T)   = p'(T)  = c1 + 2*c2*T + 3*c3*T^2 + 4*c4*T^3 + 5*c5*T^4
                a(T)   = p''(T) = 2*c2 + 6*c3*T + 12*c4*T^2 + 20*c5*T^3
        """
        State_Mat = np.array([pos0, vel0, acc0, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)
    
    # =========================== 多项式各阶导对应的物理量 =========================== #
    def get_snap(self, t):  # 四阶导
        """Return the scalar jerk at time t."""
        return 24 * self.A[4] + 120 * self.A[5] * t
    
    def get_jerk(self, t):  # 三阶导
        """Return the scalar jerk at time t."""
        return 6 * self.A[3] + 24 * self.A[4] * t + 60 * self.A[5] * t * t
    
    def get_acceleration(self, t):  # 二阶导
        """Return the scalar acceleration at time t."""
        return 2 * self.A[2] + 6 * self.A[3] * t + 12 * self.A[4] * t * t + 20 * self.A[5] * t * t * t
    
    def get_velocity(self, t):  # 一阶导
        """V: Return the scalar velocity at time t."""
        return self.A[1] + 2 * self.A[2] * t + 3 * self.A[3] * t * t + 4 * self.A[4] * t * t * t + \
               5 * self.A[5] * t * t * t * t
    
    def get_position(self, t):
        """P: Return the scalar position at time t."""
        return self.A[0] + self.A[1] * t + self.A[2] * t * t + self.A[3] * t * t * t + self.A[4] * t * t * t * t + \
               self.A[5] * t * t * t * t * t


class Polys5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """ multiple 5-th order polynomials at each Axis (only used for visualization of multiple trajectories) """
        N = len(pos1)
        State_Mat = np.array([[pos0] * N, [vel0] * N, [acc0] * N, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)
    
    def get_position(self, t):
        """Return the position array at time t."""
        t = np.atleast_1d(t)
        result = (self.A[0][:, np.newaxis] + self.A[1][:, np.newaxis] * t + self.A[2][:, np.newaxis] * t ** 2 +
                  self.A[3][:, np.newaxis] * t ** 3 + self.A[4][:, np.newaxis] * t ** 4 + self.A[5][:,
                                                                                          np.newaxis] * t ** 5)
        return result.flatten()


def wrap_to_pi(angle):
    """将角度限制在 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_yaw(vel_dir, goal_dir, last_yaw, dt, max_yaw_rate=0.5):
    # Normalize velocity and goal directions
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-5)
    goal_dist = np.linalg.norm(goal_dir)
    goal_dir = goal_dir / (goal_dist + 1e-5)
    
    # Goal yaw and weighting
    goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
    delta_yaw = wrap_to_pi(goal_yaw - last_yaw)
    weight = 6 * abs(delta_yaw) / np.pi  # weight ∈ [0,6]; equal weight at 30°, goal weight increases as delta_yaw grows
    
    # Desired direction and yaw
    dir_des = vel_dir + weight * goal_dir
    yaw_desired = np.arctan2(dir_des[1], dir_des[0]) if goal_dist > 0.5 else last_yaw
    
    # Yaw difference and limit
    yaw_diff = wrap_to_pi(yaw_desired - last_yaw)
    max_yaw_change = max_yaw_rate * np.pi * dt
    yaw_change = np.clip(yaw_diff, -max_yaw_change, max_yaw_change)
    
    # Updated yaw and yaw rate
    yaw = wrap_to_pi(last_yaw + yaw_change)
    yawdot = yaw_change / dt
    
    return yaw, yawdot


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    solver = Poly5Solver(0, 0, 0, 10, 0, 0, 2)
    t = np.linspace(0, 2, 100)
    pos = [solver.get_position(ti) for ti in t]
    vel = [solver.get_velocity(ti) for ti in t]
    acc = [solver.get_acceleration(ti) for ti in t]
    jerk = [solver.get_jerk(ti) for ti in t]
    
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t, pos)
    plt.ylabel('position')
    plt.subplot(4, 1, 2)
    plt.plot(t, vel)
    plt.ylabel('velocity')
    plt.subplot(4, 1, 3)
    plt.plot(t, acc)
    plt.ylabel('acceleration')
    plt.subplot(4, 1, 4)
    plt.plot(t, jerk)
    plt.ylabel('jerk')
    plt.xlabel('time')
    plt.show()
