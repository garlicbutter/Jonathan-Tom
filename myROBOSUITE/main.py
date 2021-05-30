from my_environment import MyEnv
from my_controller import controller_config
import numpy as np
from motion_planning_class import Policy_action

if __name__ == "__main__":
	# Task configuration
	# option:
	# 			board	: Hole 12mm, Hole 9mm
	#			peg		: 16mm. 12mm, 9mm
	#			USB		: USB-C
	task_config = {'board': 'GMC',
					'peg' : '16mm'}

	# create environment instance
	env = MyEnv(robots="UR5e",
				task_configs=task_config,
				controller_configs=controller_config,
				has_renderer=True,
				has_offscreen_renderer=False,
				use_camera_obs=False,
				render_camera=None,
				ignore_done=True)
	
	# initilaize motion planning class
	motion_planning = Policy_action()

	# define useful variables
	dof = env.robots[0].dof
    # Initial action
	action = np.zeros(dof)
	# done = True when the task is completed
	done = False
	# eef_pos_history for integrating
	eef_pos_history = np.array([])
	# action status
	action_status = {'moved_to_object':False,
					 'raised'		  :False,
					 'grabbed'		  :False}

	# manual control via keyboard
	# Keys            Command
	# q               reset simulation
	# spacebar        toggle gripper (open/close)
	# w-a-s-d         move arm horizontally in x-y plane
	# r-f             move arm vertically
	# z-x             rotate arm about x-axis
	# t-g             rotate arm about y-axis
	# c-v             rotate arm about z-axis
	# ESC             quit
	manual_control = False

	# initialize device
	if manual_control:
		from robosuite.devices import Keyboard
		from robosuite.utils.input_utils import input2action
		device = Keyboard(pos_sensitivity=0.2, rot_sensitivity=1.0)
		env.viewer.add_keypress_callback("any", device.on_press)
		env.viewer.add_keyup_callback("any", device.on_release)
		env.viewer.add_keyrepeat_callback("any", device.on_press)

	while not done:
		obs, reward, done, _ = env.step(action)	# take action in the environment
		# action, action_status = get_policy_action(obs, action_status)         # use observation to decide on an action

		if manual_control:
			# Get the newest action
			action, grasp = input2action(
				device=device,
				robot=env.robots[0],
			)
		else:
			action, action_status = motion_planning.get_policy_action(obs, action_status, env.control_timestep)         # use observation to decide on an action

		env.render()  							# render






		

