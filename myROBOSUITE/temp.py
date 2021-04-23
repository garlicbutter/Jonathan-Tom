import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import UR5e
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint

# Creating the world
world = MujocoWorldBase()

# Creating the robot
mujoco_robot = UR5e()

# add a gripper to the robot by creating a gripper instance and calling the add_gripper method on a robot
gripper = gripper_factory('Robotiq85Gripper')
mujoco_robot.add_gripper(gripper)

# Creating the table
mujoco_arena = TableArena()
table_coord = [0.5, 0, 0]
mujoco_arena.set_origin(table_coord)
world.merge(mujoco_arena)


# place the robot on to a desired position and merge it into the world
robot_base_coord = mujoco_arena.table_top_abs
mujoco_robot.set_base_xpos(robot_base_coord)
world.merge(mujoco_robot)


# Running Simulation
mymodel = world.get_model(mode="mujoco_py")


