import numpy as np
from robosuite.utils.transform_utils import quat2mat

def get_policy_action(obs, action_status, dt ,eef_pos_history):
    # Initial action
    action = np.zeros(7)
    moved_to_object = action_status['moved_to_object']
    grabbed = action_status['grabbed']
    raised = action_status['raised']

    # Observed value
    eef_pos 		= obs['robot0_eef_pos'] 					# array 1x3
    eef_quat 		= obs['robot0_eef_quat'] 					# array 1x4
    eef_quat[0],eef_quat[1],eef_quat[2]		 = eef_quat[1],eef_quat[0],-eef_quat[2]
    eef_vel         = obs['robot0_gripper_qvel'][0:3]

    peg_pos 		= obs['peg_pos'] + np.array([0,0,0.025])	# array 1x3
    peg_quat 		= obs['peg_quat']							# array 1X4
    eef_to_peg_pos 	= peg_pos - eef_pos							# array 1x3
    eef_to_peg_quat = peg_quat - eef_quat						# array 1x3

    hole_pos 		= obs['plate_pos'] + quat2mat(obs['plate_quat']) @ np.array([0.155636,0.1507,0.1])	# array 1x3
    eef_to_hole_pos = hole_pos - eef_pos						# array 1x3

    # integration of eef error (for PID)
    if len(eef_pos_history) > 1 : # in case it's a empty array
        eef_pos_history = eef_pos_history.reshape((-1,3))  #  we need this for integration
        eef_pos_error_history = peg_pos - eef_pos_history
        eef_pos_total_error   = np.sum(eef_pos_error_history, axis = 0) * dt
    else:
        action, action_done = np.array([0,0,0,0,0,0,0]), False
        action_status = {'moved_to_object':moved_to_object,
					'raised':raised,
					'grabbed':grabbed}
        return action, action_done, action_status

    if np.linalg.norm(eef_to_peg_pos) > 0.002 and not grabbed:
        x_action = 2 * eef_to_peg_pos + 0.4 * eef_pos_total_error 
        w_action = 0.1*np.array([0,0,eef_to_peg_quat[2],0])
        action = np.concatenate((x_action,w_action))
        action_done = False
        action_status = {'moved_to_object':moved_to_object,
					'raised':raised,
					'grabbed':grabbed}        
        return action, action_done, action_status
     
    elif np.linalg.norm(eef_to_peg_pos) < 0.002 and not grabbed:
        action = np.concatenate( (eef_to_peg_pos, np.array([0,0,0,1]) ) ) 
        grabbed = True
        action_done = True
        action_status = {'moved_to_object':moved_to_object,
					'raised':raised,
					'grabbed':grabbed}
        return action, action_done, action_status

    if grabbed and not raised:
        if eef_pos[2] > 1.1:
            raised = True
            action_done = True
            action_status = {'moved_to_object':moved_to_object,
					        'raised':raised,
					        'grabbed':grabbed}
            return action, action_done, action_status
        else:
            action = np.array([0, 0, 0.2, 0, 0, 0, 1])
            action_done = False
            action_status = {'moved_to_object':moved_to_object,
                            'raised':raised,
                            'grabbed':grabbed}
            return action, action_done, action_status

    if grabbed and raised and np.linalg.norm(eef_to_hole_pos[0:2]) > 0.005:
        action = np.concatenate( (eef_to_hole_pos[0:2], np.array([0,0,0,0,1]) ) )
    elif grabbed and raised and np.linalg.norm(eef_to_hole_pos[0:2]) < 0.005:
        action = np.array([0, 0, -0.1, 0, 0, 0, 1])
