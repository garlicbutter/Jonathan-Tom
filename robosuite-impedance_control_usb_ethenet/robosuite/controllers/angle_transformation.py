"""
UR robot uses an axis-angle convention to describe the orientation Rx,Ry,Rz i.e rotation vector
To implement a minimum-jerk trajectory we need to convert the angle to Euler angles
Notion used is "RPY" roll-pitch-yaw convention i.e. ZYX Euler representation
http://web.mit.edu/2.05/www/Handout/HO2.PDF
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
Last modified: 14.12.2020
Daniel Stankowski
"""

from scipy.spatial.transform import Rotation as R
import numpy as np


def Robot2Euler(orient_array):
    """
    Convert axis-angle to euler zyx convention
    :param orient_array: np.array([Rx,Ry,Rz]) from the robot pose
    :return: euler angles in [rad]
    """
    orientation = np.array([orient_array[0], orient_array[1], orient_array[2]])
    temp = R.from_rotvec(orientation)
    euler = temp.as_euler("zyx", degrees=False)
    return np.array(euler)


def Euler2Robot(euler_array):
    """
    Convert euler zyx angle to axis-angle
    :param: array of euler angles in zyx convention
    :return:  np.array([Rx,Ry,Rz])
    """
    euler_test = np.array([euler_array[0], euler_array[1], euler_array[2]])
    temp2 = R.from_euler('zyx', euler_test, degrees=False)
    axis_angle = temp2.as_rotvec()
    return np.array(axis_angle)

