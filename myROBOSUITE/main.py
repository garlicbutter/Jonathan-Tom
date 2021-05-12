from my_environment import MyEnv
from my_controller import controller_config
import numpy as np

if __name__ == "__main__":
	# Task configuration
	# option:
	# 			board	: Hole 12mm, Hole 9mm
	#			peg		: 16mm. 12mm, 9mm
	#			USB		: USB-C
	task_config = {'board': 'Hole12mm',
					'peg': '16mm'}

	# create environment instance
	env = MyEnv(robots="UR5e",
				task_configs=task_config,
				controller_configs=controller_config,
				has_renderer=True,
				has_offscreen_renderer=False,
				use_camera_obs=False,
				render_camera=None)

	# define useful variables
	dof = env.robots[0].dof

	# done = True when the task is completed
	done = False

	# Initial action
	action = np.zeros(dof)
	grabbed = False

	# simulation loop
	while not done:
		obs, reward, done, _ = env.step(action)	# take action in the environment
		env.render()  							# render

		# observed value
		eef_pos 		= obs['robot0_eef_pos'] 					# array 1x3
		eef_quat 		= obs['robot0_eef_quat'] 					# array 1x4
		eef_quat[0],eef_quat[1]	 = eef_quat[1],eef_quat[0]	
		eef_quat[2]		= -eef_quat[2]

		peg_pos 		= obs['peg_pos'] + np.array([0,0,0.025])		# array 1x3
		peg_quat 		= obs['peg_quat']							# array 1X4
		eef_to_peg_pos 	= peg_pos - eef_pos							# array 1x3
		eef_to_peg_quat = peg_quat - eef_quat						# array 1x3
		# dt				= env.control_timestep
		
		if grabbed:
			action = np.array([0,0,0.2,0,0,0,1])

		if np.linalg.norm(eef_to_peg_pos) > 0.01 and not grabbed:
			action = np.concatenate( (eef_to_peg_pos, 0.2*np.array([0,0,eef_to_peg_quat[2],0])) )
		elif np.linalg.norm(eef_to_peg_pos) < 0.01 and not grabbed:
			action = np.concatenate( (eef_to_peg_pos, np.array([0,0,0,1]) ) ) 
			grabbed = True





		

