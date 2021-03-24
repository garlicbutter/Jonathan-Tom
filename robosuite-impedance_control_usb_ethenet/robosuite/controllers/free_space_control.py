import numpy as np
import URBasic
import angle_transformation as at
from min_jerk_impedance import PathPlan
import time
from pd_controller import PD_controller

host = '192.168.1.103'
robotModle = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=host, robotModel=robotModle)
robot.reset_error()


desired = np.array([0.1256, -0.4317,  0.3211, -0.0425, 3.1233, 0.0984])
pos = np.array(robot.get_actual_tcp_pose())
pos[3:6] = at.Robot2Euler(pos[3:6])
total_time = 6

planner = PathPlan(pos, desired, total_time)
velocity_now = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # get_actual_tcp_speed()
t_intial = time.time()
try:
    while time.time() - t_intial < total_time:
        t = time.time() - t_intial
        pos2 = np.array(robot.get_actual_tcp_pose())
        pos2[3:6] = at.Robot2Euler(pos2[3:6])

        [position, orientation, velocity, ang_vel, acceleration] = planner.trajectory_planning(t)

        delta = [position - pos2[:3], orientation - pos2[3:], velocity - velocity_now[:3], ang_vel - velocity_now[3:]]
        wrench_task = PD_controller(delta)

        robot.set_force_remote(task_frame=[0, 0, 0, 0, 0, 0], selection_vector=[1, 1, 1, 1, 1, 1],
                               wrench=list(wrench_task), f_type=2, limits=[2, 2, 1.5, 1, 1, 1])

        velocity_now = np.array(robot.get_actual_tcp_speed())

except Exception as e:
    print("error in first section")
    print(e)
    robot.end_force_mode()
    robot.reset_error()
    robot.close()


