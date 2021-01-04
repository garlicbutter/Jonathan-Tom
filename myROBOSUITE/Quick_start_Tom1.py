# Step 1: Creating the world. All mujoco object definitions are housed in an xml. We create a MujocoWorldBase class to do it.

from robosuite.models import MujocoWorldBase

world = MujocoWorldBase()


# Step 2: Creating the table. We can initialize the TableArena instance that creates a table and the floorplane

from robosuite.models.arenas import TableArena


mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)
# self.table_full_size = table_full_size
# self.table_friction = table_friction
# self.table_offset = np.array((0, 0, 0.8))

# Step 3: Creating the robot. The class housing the xml of a robot can be created as follows.

from robosuite.models.robots import UR5e

mujoco_robot = UR5e()

# We can add a gripper to the robot by creating a gripper instance and calling the add_gripper method on a robot.

from robosuite.models.grippers import gripper_factory

gripper = gripper_factory('Robotiq85Gripper')
mujoco_robot.add_gripper(gripper)

# To add the robot to the world, we place the robot on to a desired position and merge it into the world


xpos = table_offset.tolist()
mujoco_robot.set_base_xpos(xpos)
        # self.table_full_size = table_full_size
        # self.table_friction = table_friction
        # self.table_offset = np.array((0, 0, 0.8))
# mujoco_robot.set_base_xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
# self.robots[0].robot_model.set_base_xpos(xpos)
world.merge(mujoco_robot)



# Step 4: Adding the object. 
# For details of MujocoObject, refer to the documentations about MujocoObject, we can create a ball and add it to the world. 
# It is a bit more complicated than before because we are adding a free joint to the object (so it can move) and we want to place the object properly.

from robosuite.models.objects import CylinderObject
from robosuite.utils.mjcf_utils import new_joint

cylinder = CylinderObject(
    name="cylinder",
    size=[0.04, 2],
    rgba=[2, 2, 2, 2])
#sphere.append(new_joint(name='sphere_free_joint', type='free'))
# cylinder.set('pos', '1.0 0 1.0')
#world.worldbody.append(cylinder)


# Step 5: Running Simulation. Once we have created the object, we can obtain a mujoco_py model by running
model = world.get_model(mode="mujoco_py")

# This is an MjModel instance that can then be used for simulation. For example,

from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
  sim.data.ctrl[:] = 0
  sim.step()
  viewer.render()

