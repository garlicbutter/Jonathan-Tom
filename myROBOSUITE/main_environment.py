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
gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)

# place the robot on to a desired position and merge it into the world
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

# Creating the table
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

# Running Simulation
mymodel = world.get_model(mode="mujoco_py")
