import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import UR5e
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_joint
from mujoco_py import MjSim, MjViewer


world = MujocoWorldBase()
mujoco_robot = UR5e()
gripper = gripper_factory('JacoThreeFingerGripper')
mujoco_robot.add_gripper(gripper)
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

box1 = BoxObject(
    name="Box",
    size=[0.05, 0.08, 0.03],
    rgba=[0, 0.5, 0.5, 1],
    density=1,
    joints="default",
    duplicate_collision_geoms=True)

box1.set('pos', '1.0 0 1.0')
world.worldbody.append(box1)

model = world.get_model(mode="mujoco_py")

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0  # disable visualization of collision mesh

for i in range(10000):
    sim.data.ctrl[:] = 0
    sim.step()
    viewer.render()


# Load the desired controller's default config as a dict
# controller_config = load_controller_config(default_controller="JOINT_VELOCITY")


# create environment instance
# env = suite.make(
#     env_name="Lift",  # try with other tasks like "Stack" and "Door"
#     robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
#     controller_configs=controller_config,
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
# )


# reset the environment
# env.reset()

# for i in range(1000):
#     action = np.random.randn(env.robots[0].dof)  # sample random action
#     # take action in the environment
#     obs, reward, done, info = env.step(action)
#     env.render()  # render on display
