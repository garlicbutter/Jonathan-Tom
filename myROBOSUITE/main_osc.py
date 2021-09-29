import numpy as np
from my_environment import MyEnv
from my_environment_offscreen import MyEnvOffScreen
import motion_planning_osc
import matplotlib.pyplot as plt

def main_osc(controller_kp = [1500, 1500, 50, 150, 150, 150],
			controller_zeta = [1.5, 1.5, 1, 10, 10, 10],
			perception_error = 0.0,
			offscreen = False):
	# Task configuration
	# option:
	# 			board	: Hole 12mm, Hole 9mm
	#			peg		: 16mm. 12mm, 9mm
	#			USB		: USB-C
	task_config = {'board': 'Square_hole_16mm',
					'peg' : 'cylinder_16mm'}
	# IMP_OSC is a custom controller written by us.
	# find out the source code at https://github.com/garlicbutter/robosuite
	# Theory based on the paper by Valency: https://doi.org/10.1115/1.1590685.



	controller_config = {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                    "kp": controller_kp,
                    "damping_ratio": controller_zeta,
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
	if offscreen:
		# create on-screen environment instance
		env = MyEnvOffScreen(robots="UR5e",
							task_configs=task_config,
							controller_configs=controller_config,
							gripper_types='Robotiq85Gripper',
							ignore_done=True)
	else:
		# create on-screen environment instance
		env = MyEnv(robots="UR5e",
					task_configs=task_config,
					controller_configs=controller_config,
					gripper_types='Robotiq85Gripper',
					has_renderer=True,
					has_offscreen_renderer=False,
					use_object_obs=True,
					use_camera_obs=False,
					ignore_done=True,
					render_camera=None)
					
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
	
	# Initial recorder
	eeff_record = np.empty([1,3]) # end effector force record
	robot_torque_record = np.empty([1,6]) # Actuation torque record
	eeft_record = np.empty([1,3]) # end effector torque record
	eefd_record = np.empty([1,3]) # end effector displacement record
	eefd_desire_record = np.empty([1,3]) # desired end effector displacement record
	eefd_desire = np.empty([1,3]) # to calculate desired displacement at each moment
	t_record = np.empty(1) # time record

	# Initial action
	action = np.zeros(env.robots[0].dof)
	action_status = None
	# simulate termination condition
	done = False
	# simulation

	# for perception_error
	import random
	random_radiant = 2*np.pi*random.random()
	perception_err_x = np.cos(random_radiant) * perception_error
	perception_err_y = np.sin(random_radiant) * perception_error


	while not done:
		obs, reward, done, _ = env.step(action)	# take action in the environment
		if manual_control:
			action, grasp = input2action(
				device=device,
				robot=env.robots[0],
			)
		else:
			obs['plate_pos'][0] += perception_err_x
			obs['plate_pos'][1] += perception_err_y

			# update observation to motion planning11
			motion_planning.update_obs(obs)
			# decide which action to take for next simulation
			action, action_status = motion_planning.get_policy_action()    

		if not offscreen:
			env.render()
	
		eeff_record = np.append(eeff_record,[env.robots[0].ee_force],axis=0)
		eeft_record = np.append(eeft_record,[env.robots[0].ee_torque],axis=0)
		eefd_record = np.append(eefd_record,[obs['robot0_eef_pos']],axis=0)
		eefd_desire = np.add(eefd_desire,action[0:3])
		eefd_desire_record = np.append(eefd_desire_record,eefd_desire,axis=0)
		robot_torque_record = np.append(robot_torque_record, [env.robots[0].torques], axis=0)
		t_record = np.append(t_record,np.array([env.cur_time]),axis=0)
		
		if action_status['done']:
			break
		if t_record[-1] > 30: # failure case
			eeff_record = 0
			eeft_record = 0
			eefd_record = 0
			t_record	= 0
			break
	return eeff_record, eeft_record, eefd_record, t_record, robot_torque_record

def plotter(eeff_record, eeft_record, eefd_record, t_record):
	# plot of end effector force record 
	fig_eeff, ax_eeff = plt.subplots(3)
	fig_eeff.figsize=(10,6)
	fig_eeff.suptitle('End_Effector Force v.s. Time')
	ylabels_eeff = [r'$F_x [N]$',r'$F_y [N]$',r'$F_z [N]$']
	for idx,ylabel_eeff in enumerate(ylabels_eeff):
		ax_eeff[idx].grid()
		ax_eeff[idx].set_ylabel(ylabel_eeff)
		ax_eeff[idx].set_xlabel(r't[s]')
		ax_eeff[idx].plot(t_record,eeff_record[:,idx])
	# plot of end effector torque record
	fig_eeft, ax_eeft = plt.subplots(3)
	fig_eeft.figsize=(10,6)
	fig_eeft.suptitle('End-Effector Torque v.s. Time')
	ylabels_eeft = [r'$T_x [N]$',r'$T_y [N]$',r'$T_z [N]$']
	for idx,ylabel_eeft in enumerate(ylabels_eeft):
		ax_eeft[idx].grid()
		ax_eeft[idx].set_ylabel(ylabel_eeft)
		ax_eeft[idx].set_xlabel(r't[s]')
		ax_eeft[idx].plot(t_record,eeft_record[:,idx])
	# plot of end effector displacement record
	fig_eefd, ax_eefd = plt.subplots(3)
	fig_eefd.figsize=(10,6)
	fig_eefd.suptitle('End_effector Displacement v.s. Time')
	ylabels_eefd = [r'$D_x [N]$',r'$D_y [N]$',r'$D_z [N]$']
	for idx,ylabel_eefd in enumerate(ylabels_eefd):
		ax_eefd[idx].grid()
		ax_eefd[idx].set_ylabel(ylabel_eefd)
		ax_eefd[idx].set_xlabel(r't[s]')
		ax_eefd[idx].plot(t_record,eefd_record[:,idx])
		# ax_eefd[idx].plot(t_record,eefd_desire_record[:,idx])
	plt.ioff()
	plt.show()

if __name__ == "__main__":
	eeff_record, eeft_record, eefd_record, t_record, robot_torque_record = main_osc(controller_kp = [1500, 1500, 50, 150, 150, 150],
															   controller_zeta = [1.5, 1.5, 1, 10, 10, 10],
															   perception_error = 0.0,
															   offscreen=False)
	plotter(eeff_record, eeft_record, eefd_record, t_record)

	






