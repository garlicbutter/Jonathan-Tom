"""
Minimum jerk trajectory for 6DOF robot
Latest update: 14.12.2020
Written by Daniel Stankowski
"""
import numpy as np
from matplotlib import pyplot as plt
import angle_transformation as at


class PathPlan(object):
    """
    IMPORTANT: When the pose is passed [x,y,z,Rx,Ry,Rz] one has to convert the orientation
    part from axis-angle representation to the Euler before running this script.
    """

    def __init__(self, initial_point, target_point, total_time):
        self.initial_position = initial_point[:3]
        self.target_position = target_point[:3]
        self.initial_orientation = initial_point[3:]
        self.target_orientation = target_point[3:]
        self.tfinal = total_time

    def trajectory_planning(self, t):
        X_init = self.initial_position[0]
        Y_init = self.initial_position[1]
        Z_init = self.initial_position[2]

        X_final = self.target_position[0]
        Y_final = self.target_position[1]
        Z_final = self.target_position[2]

        slope_y = (Y_final - Y_init) / (X_final - X_init + 1e-6)
        b_y = (X_final * Y_init - X_init * Y_final) / (X_final - X_init + 1e-6)
        slope_z = (Z_final - Z_init) / (X_final - X_init + 1e-6)
        b_z = (X_final * Z_init - X_init * Z_final) / (X_final - X_init + 1e-6)

        # position
        x_traj = (X_final - X_init) / (self.tfinal ** 3) * (
                    6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + X_init
        y_traj = slope_y * x_traj + b_y
        z_traj = slope_z * x_traj + b_z
        position = np.array([x_traj, y_traj, z_traj])

        # velocities
        vx = (X_final - X_init) / (self.tfinal ** 3) * (
                    30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        vy = slope_y * vx
        vz = slope_z * vx
        velocity = np.array([vx, vy, vz])

        # acceleration
        ax = (X_final - X_init) / (self.tfinal ** 3) * (
                    120 * (t ** 3) / (self.tfinal ** 2) - 180 * (t ** 2) / self.tfinal + 60 * t)
        ay = slope_y * ax
        az = slope_z * ax
        acceleration = np.array([ax, ay, az])

        #   -----------------------------------rotation (given in Euler form) ---------------------------------------
        alpha_init = self.initial_orientation[0]
        beta_init = self.initial_orientation[1]
        gamma_init = self.initial_orientation[2]

        alpha_final = self.target_orientation[0]
        beta_final = self.target_orientation[1]
        gamma_final = self.target_orientation[2]

        slope_beta = (beta_final - beta_init) / (alpha_final - alpha_init + 1e-6)
        b_beta = (alpha_final * beta_init - alpha_init * beta_final) / (alpha_final - alpha_init + 1e-6)
        slope_gamma = (gamma_final - gamma_init) / (alpha_final - alpha_init + 1e-6)
        b_gamma = (alpha_final * gamma_init - alpha_init * gamma_final) / (alpha_final - alpha_init + 1e-6)

        # orientation
        alpha_traj = (alpha_final - alpha_init) / (self.tfinal ** 3) * (
                    6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + alpha_init
        beta_traj = slope_beta * alpha_traj + b_beta
        gamma_traj = slope_gamma * alpha_traj + b_gamma
        orientation = np.array([alpha_traj, beta_traj, gamma_traj])

        # angular velocities
        alpha_d_traj = (alpha_final - alpha_init) / (self.tfinal ** 3) * (
                    30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        beta_d_traj = slope_beta * alpha_d_traj
        gamma_d_traj = slope_gamma * alpha_d_traj
        ang_vel = np.array([alpha_d_traj, beta_d_traj, gamma_d_traj])

        return [position, orientation, velocity, ang_vel, acceleration]

    # ***** MAIN CODE **********


if __name__ == "__main__":
    initial_point = np.array([-0.1779, -0.4952, 0.2933, 3.1267, -0.0438, 0.0142])
    target_point = np.array([0.1279, -0.5259,  0.1048,  3.1267, -0.0439,  0.0142])
    initial_point[3:6] = at.Robot2Euler(initial_point[3:6])
    target_point[3:6] = at.Robot2Euler(target_point[3:6])
    tfinal = 6.5

    trajectory = PathPlan(initial_point, target_point, tfinal)
    posx = []
    posy = []
    posz = []
    v_x = []
    v_y = []
    v_z = []
    a_x = []
    a_y = []
    a_z = []
    j_x = []
    j_y = []
    j_z = []
    ox = []
    oy = []
    oz = []
    avx = []
    avy = []
    avz = []
    time_range = []

    for i in range(100):
        t = (i / 100) * tfinal
        [position, orientation, velocity, ang_vel, acceleration] = trajectory.trajectory_planning(t)
        # orientation
        ox.append(orientation[0])
        oy.append(orientation[1])
        oz.append(orientation[2])
        # angular velocity
        avx.append(ang_vel[0])
        avy.append(ang_vel[1])
        avz.append(ang_vel[2])
        # position
        posx.append(position[0])
        posy.append(position[1])
        posz.append(position[2])
        # velocity
        v_x.append(velocity[0])
        v_y.append(velocity[1])
        v_z.append(velocity[2])
        # acceleration
        a_x.append(acceleration[0])
        a_y.append(acceleration[1])
        a_z.append(acceleration[2])

        time_range.append(t)

    # plotting using pyplot
    plt.figure()
    plt.plot(time_range, posx, label='X position')
    plt.plot(time_range, posy, label='Y position')
    plt.plot(time_range, posz, label='Z position')
    plt.legend()
    plt.grid()
    plt.ylabel('Position [m]')
    plt.xlabel('Time [s]')
    plt.show()

    plt.figure()
    plt.plot(time_range, v_x, label='X velocity')
    plt.plot(time_range, v_y, label='Y velocity')
    plt.plot(time_range, v_z, label='Z velocity')
    plt.legend()
    plt.grid()
    plt.ylabel('Velocity[m/s]')
    plt.xlabel('Time [s]')
    plt.show()

    plt.figure()
    plt.plot(time_range, a_x, label='X acc')
    plt.plot(time_range, a_y, label='Y acc')
    plt.plot(time_range, a_z, label='Z acc')
    plt.legend()
    plt.grid()
    plt.ylabel('Acceleration [m/s^2]')
    plt.xlabel('Time [s]')
    plt.show()

    plt.figure()
    plt.plot(time_range, ox, label='Rx orientation')
    plt.plot(time_range, oy, label='Ry orientation')
    plt.plot(time_range, oz, label='Rz orientation')
    plt.legend()
    plt.grid()
    plt.ylabel('Orientation')
    plt.xlabel('Time [s]')
    plt.show()

    plt.figure()
    plt.plot(time_range, avx, label='Rx ang velocity')
    plt.plot(time_range, avy, label='Ry ang velocity')
    plt.plot(time_range, avz, label='Rz ang velocity')
    plt.legend()
    plt.grid()
    plt.ylabel('Angular Velocity')
    plt.xlabel('Time [s]')
    plt.show()
