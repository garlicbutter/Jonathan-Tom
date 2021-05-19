import numpy as np
from robosuite.utils.transform_utils import quat2mat

def get_policy_action(obs):
    # Initial action
    action = np.zeros(7)
    moved_to_object = False
    grabbed = False
    raised = False

    # observed value
    eef_pos 		= obs['robot0_eef_pos'] 					# array 1x3
    eef_quat 		= obs['robot0_eef_quat'] 					# array 1x4
    eef_quat[0],eef_quat[1]	 = eef_quat[1],eef_quat[0]	
    eef_quat[2]		= -eef_quat[2]

    peg_pos 		= obs['peg_pos'] + np.array([0,0,0.025])	# array 1x3
    peg_quat 		= obs['peg_quat']							# array 1X4
    eef_to_peg_pos 	= peg_pos - eef_pos							# array 1x3
    eef_to_peg_quat = peg_quat - eef_quat						# array 1x3

    hole_pos 		= obs['plate_pos'] + quat2mat(obs['plate_quat']) @ np.array([0.155636,0.1507,0.1])	# array 1x3
    eef_to_hole_pos = hole_pos - eef_pos						# array 1x3

    # dt				= env.control_timestep

    if np.linalg.norm(eef_to_peg_pos) > 0.002 and not grabbed:
        action = np.concatenate( (1.5*eef_to_peg_pos, 0.2*np.array([0,0,eef_to_peg_quat[2],0])) )
        
    elif np.linalg.norm(eef_to_peg_pos) < 0.002 and not grabbed:
        action = np.concatenate( (eef_to_peg_pos, np.array([0,0,0,1]) ) ) 
        grabbed = True

    if grabbed and not raised:
        action = np.array([0, 0, 0.2, 0, 0, 0, 1])
        if eef_pos[2] > 1.1:
            raised = True

    if grabbed and raised and np.linalg.norm(eef_to_hole_pos[0:2]) > 0.005:
        action = np.concatenate( (eef_to_hole_pos[0:2], np.array([0,0,0,0,1]) ) )
    elif grabbed and raised and np.linalg.norm(eef_to_hole_pos[0:2]) < 0.005:
        action = np.array([0, 0, -0.1, 0, 0, 0, 1])

    return action