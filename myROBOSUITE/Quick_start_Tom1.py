from robosuite.models import MujocoWorldBase

world = MujocoWorldBase()

from robosuite.models.robots import Panda

mujoco_robot = Panda()

from robosuite.models.grippers import gripper_factory

gripper = gripper_factory('PandaGripper')
# gripper.hide_visualization()
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint

# sphere = BallObject(
#     name="sphere",
#     size=[0.04],
#     # rgba=[0, 0.5, 0.5, 1]).get_collision()
# sphere.append(new_joint(name='sphere_free_joint', type='free'))
# sphere.set('pos', '1.0 0 1.0')
# world.worldbody.append(sphere)



model = world.get_model(mode="mujoco_py")

from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
  sim.data.ctrl[:] = 0
  sim.step()
  viewer.render()