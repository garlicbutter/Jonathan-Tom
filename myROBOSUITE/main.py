from mujoco_py import MjSim, MjViewer
from main_environment import mymodel

sim = MjSim(mymodel)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
  sim.data.ctrl[:] = 0
  sim.step()
  viewer.render()