import numpy as np
import os
from my_environment import MyEnv
import motion_planning_osc

if __name__ == "__main__":
	# Task configuration
	# option:
	# 			board	: Hole 12mm, Hole 9mm
	#			peg		: 16mm. 12mm, 9mm
	#			USB		: USB-C
	task_config = {'board': 'Square_hole_16mm',
					'peg' : 'cylinder_16mm'}
	# IMP_OSC is a custom controller written by us.
	# find out the source code at https://github.com/garlicbutter/robosuite
	# Theory based on the paper by Valency: https://doi.org/10.1115/1.1590685
	controller_config = {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": [150, 150, 150, 150, 150, 150],
                    "damping_ratio": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [[0, 300], [0, 300], [0, 300], [0, 300], [0, 300], [0, 300]],
                    "damping_ratio_limits": [0, 10],
                    "position_limits": None,
                    "orientation_limits": None,
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": None,
                    "ramp_ratio": 0.2
                    }
	# create environment instance
	env = MyEnv(robots="UR5e",
				task_configs=task_config,
				controller_configs=controller_config,
				gripper_types='Robotiq85Gripper',
				has_renderer=True,
				has_offscreen_renderer=False,
				use_camera_obs=False,
				render_camera=None,
				ignore_done=True)
	# create motion planning class
	motion_planning = motion_planning_osc.Policy_action(env.control_timestep,
														P=3,
														I=0.1)
	# manual control via keyboard
	manual_control = False
	if manual_control:
		from robosuite.devices import Keyboard
		from robosuite.utils.input_utils import input2action
		device = Keyboard(pos_sensitivity=0.05, rot_sensitivity=1.0)
		env.viewer.add_keypress_callback("any", device.on_press)
		env.viewer.add_keyup_callback("any", device.on_release)
		env.viewer.add_keyrepeat_callback("any", device.on_press)
    # Initial action
	action = np.zeros(env.robots[0].dof)
	action_status = None
	# simulate termination condition
	done = False
	# simulation  
	while not done:
		obs, reward, done, _ = env.step(action)	# take action in the environment
		if manual_control:
			action, grasp = input2action(
				device=device,
				robot=env.robots[0],
			)
		else:
			# obs['plate_pos'][0] += 0.0016
			# obs['plate_pos'][1] += 0.0016

			# update observation to motion planning11
			motion_planning.update_obs(obs)
			# decide which action to take for next simulation
			action, action_status = motion_planning.get_policy_action()    

		env.render()
		
		os.system('clear')
		print("Robot: {}, Gripper:{}".format(env.robots[0].name,env.robots[0].gripper_type))
		print("Control Frequency:{}".format(env.robots[0].control_freq))
		print("Action status:{}".format(action_status))
		print("eef_force:\n \t x: {a[0]:2.4f}, y: {a[1]:2.4f}, z: {a[2]:2.4f}".format(a=env.robots[0].ee_force))
		print("eef_torque:\n \t x: {a[0]:2.4f}, y: {a[1]:2.4f}, z: {a[2]:2.4f}".format(a=env.robots[0].ee_torque))