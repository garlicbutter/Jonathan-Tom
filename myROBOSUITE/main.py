from my_environment import MyEnv
from my_controller import controller_config
import numpy as np

if __name__ == "__main__":
	# Task configuration
	# option:
	# 			board	: Hole 12mm, Hole 9mm
	#			peg		: 12mm, 9mm
	#			USB		: USB-C
	task_config = {'board': 'Hole12mm',
					'peg': '12mm'}

	# create environment instance
	env = MyEnv(robots="UR5e",
				task_configs=task_config,
				controller_configs=controller_config,
				has_renderer=True,
				has_offscreen_renderer=False,
				use_camera_obs=False,
				render_camera=None)

	# reset the environment
	env.reset()

	# define useful variables
	dof = env.robots[0].dof

	# done = True when the task is completed
	done = False

	# simulation loop
	while not done:
		# action = np.zeros(dof) 					
		action = np.array([0, 0, 0, 0, 0, 0, 0])
		obs, reward, done, info = env.step(action)  # take action in the environment
		env.render()  								# render