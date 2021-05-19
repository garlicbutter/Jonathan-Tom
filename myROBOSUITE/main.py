from my_environment import MyEnv
from my_controller import controller_config
import numpy as np
from motion_planning import get_policy_action

if __name__ == "__main__":
	# Task configuration
	# option:
	# 			board	: Hole 12mm, Hole 9mm
	#			peg		: 16mm. 12mm, 9mm
	#			USB		: USB-C
	task_config = {'board': 'GMC',
					'peg': '16mm'}

	# create environment instance
	env = MyEnv(robots="UR5e",
				task_configs=task_config,
				controller_configs=controller_config,
				has_renderer=True,
				has_offscreen_renderer=False,
				use_camera_obs=False,
				render_camera=None,
				ignore_done=True)

	# define useful variables
	dof = env.robots[0].dof
    # Initial action
	action = np.zeros(dof)
	# done = True when the task is completed
	done = False

	while not done:
		action = get_policy_action(obs)         # use observation to decide on an action
		obs, reward, done, _ = env.step(action)	# take action in the environment
		env.render()  							# render






		

