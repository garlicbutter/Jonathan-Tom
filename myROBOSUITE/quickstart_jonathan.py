import numpy as np
from mujoco_py import MjSim, MjViewer
from robosuite.models.grippers import gripper_factory
from robosuite.models import MujocoWorldBase
from robosuite.models.arenas import TableArena
from robosuite.models.robots import UR5e

# Step 1: Creating the world. All mujoco object definitions are housed in an xml. We create a MujocoWorldBase class to do it.
world = MujocoWorldBase()

# Step 2: Creating the table. We can initialize the TableArena instance that creates a table and the floorplane
# load model for table top workspace
table_full_size = (0.8, 0.8, 0.05)
table_friction = (1., 5e-3, 1e-4)
table_offset = np.array((0, 0, 0.8))

mujoco_arena = TableArena(
    table_full_size=table_full_size,
    table_friction=table_friction,
    table_offset=table_offset,
)

# Arena always gets set to zero origin
mujoco_arena.set_origin([0, 0, 0])
world.merge(mujoco_arena)


# Step 3: Creating the robot. The class housing the xml of a robot can be created as follows.
mujoco_robot = UR5e()

# We can add a gripper to the robot by creating a gripper instance and calling the add_gripper method on a robot.
gripper = gripper_factory("Robotiq85Gripper")
mujoco_robot.add_gripper(gripper)

# To add the robot to the world, we place the robot on to a desired position and merge it into the world
xpos = table_offset.tolist()
mujoco_robot.set_base_xpos(xpos)
world.merge(mujoco_robot)


# Step 4: Adding the object.
# self.cube = BoxObject(
#     name="cube",
#     size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
#     size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
#     rgba=[1, 0, 0, 1],
#     material=redwood,
# )


# Step 5: Running Simulation. Once we have created the object, we can obtain a mujoco_py model by running
model = world.get_model(mode="mujoco_py")

# This is an MjModel instance that can then be used for simulation. For example,


sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0  # disable visualization of collision mesh

for i in range(10000):
    sim.data.ctrl[:] = 0
    sim.step()
    viewer.render()
