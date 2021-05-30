import numpy as np
from robosuite.utils.transform_utils import quat2mat

class Policy_action:
    def __init__(
        self,
    ):
        self.eef_pos_error_int = 0

    def get_policy_action(self, obs, action_status, dt):
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

        self.eef_pos_error_int += eef_to_peg_pos*dt

        if np.linalg.norm(eef_to_peg_pos) > 0.002 and not grabbed:
            # self.eef_pos_error_int = 0
            x_action = 1 * eef_to_peg_pos + 0.01 * self.eef_pos_error_int 
            w_action = 0.01*np.array([0,0,eef_to_peg_quat[2],0])
            action = np.concatenate((x_action,w_action))
            action_status = {'moved_to_object':moved_to_object,
                            'raised':raised,
                            'grabbed':grabbed}        
            return action, action_status
        
        elif np.linalg.norm(eef_to_peg_pos) < 0.2 and not grabbed:
            action = np.concatenate( (eef_to_peg_pos, np.array([0,0,0,1]))) 
            grabbed = True
            self.eef_pos_error_int = 0 # renew the intergration to 0 for new conduction
            action_status = {'moved_to_object':moved_to_object,
                        'raised':raised,
                        'grabbed':grabbed}
            return action, action_status

        if grabbed and not raised:
            if eef_pos[2] > 1.1:
                raised = True
                self.eef_pos_error_int = 0 # renew the intergration to 0 for new conduction
                action_status = {'moved_to_object':moved_to_object,
                                'raised':raised,
                                'grabbed':grabbed}
                return action, action_status
            else:
                action = np.array([0, 0, 0.2, 0, 0, 0, 1])
                action_status = {'moved_to_object':moved_to_object,
                                'raised':raised,
                                'grabbed':grabbed}
                return action, action_status

        if grabbed and raised and np.linalg.norm(eef_to_hole_pos[0:2]) > 0.005:
            action = np.concatenate( (eef_to_hole_pos[0:2], np.array([0,0,0,0,1]) ) )
            return action, action_status
        elif grabbed and raised and np.linalg.norm(eef_to_hole_pos[0:2]) < 0.005:
            action = np.array([0, 0, -0.1, 0, 0, 0, 1])
            return action, action_status
