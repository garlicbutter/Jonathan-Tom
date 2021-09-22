import numpy as np
from numpy.linalg import norm
from robosuite.utils.transform_utils import quat2mat

class Policy_action:
    '''
    take obs as input
    return motion as output
    '''
    def __init__(
        self,
        dt,
        P=1,
        I=0.1,
    ):
        # Initiallize Observed value
        self.eef_pos_overall_error = 0
        self.eef_pos 		    =   None					
        self.eef_quat 		    =   None				
        self.eef_vel            =   None
        self.peg_pos 		    =   None 
        self.peg_quat 		    =   None
        self.eef_to_peg_pos     =   None
        self.eef_to_peg_quat    =   None
        self.hole_pos 		    =   None
        self.eef_to_hole_pos    =   None
        self.dt                 =   dt
        self.motion_P           =   P
        self.motion_I           =   I
        # action status
        self.action_status      ={'moved_to_object':False,
                                'grabbed'		  :False,
                                'moved_to_target':False,
                                'done':False}
        self.grab_wait_counter = 10

    def update_obs(self,obs):
        # Update Observed value
        self.eef_pos 		=   obs['robot0_eef_pos'] 					# array 1x3
        self.eef_quat 		=   obs['robot0_eef_quat'] 					# array 1x4
        self.eef_quat[0]    =   self.eef_quat[1]
        self.eef_quat[1]    =   self.eef_quat[0]
        self.eef_quat[2]    =   -self.eef_quat[2]
        self.eef_vel        =   obs['robot0_gripper_qvel'][0:3]
        self.peg_pos 		=   obs['peg_pos'] + quat2mat(obs['peg_quat']) @ np.array([0,0,0.015])	# array 1x3
        self.peg_quat 		=   obs['peg_quat']							# array 1X4
        self.eef_to_peg_pos =   self.peg_pos - self.eef_pos							# array 1x3
        self.eef_to_peg_quat = self.peg_quat - self.eef_quat						# array 1x3
        self.hole_pos 		= obs['plate_pos'] #+ quat2mat(obs['plate_quat']) @ np.array([0.155636,0.1507,0.1])	# array 1x3
        self.eef_to_hole_pos = self.hole_pos - self.eef_pos						# array 1x3
        self.hole_to_peg_pos = self.peg_pos - self.hole_pos						# array 1x3
        # decide status based on the observation
        self.decide_status()


    def decide_status(self):
        '''
        decide which status the robot is in.
        action status: {"moved_to_object":True/False,
                        "grabbed":True/False,
                        "moved_to_target":True/False}

        moved_to_object : if the eef position is near the object.
        grabbed         : if the eef grabbed the object.
        moved_to_target : if the eef moved the object near the target
        '''
        if norm(self.eef_to_peg_pos) < 0.002:
            self.action_status['moved_to_object'] = True

        # if self.action_status['moved_to_object'] and not self.action_status['grabbed']:
        #     self.action_status['grabbed'] = True

        if self.action_status['grabbed'] and norm(self.eef_to_hole_pos[0:2]) < 0.002:
            self.action_status['moved_to_target'] = True

        return self.action_status

    def get_policy_action(self):
        '''
        take obs as input
        return motion as output
        '''
        # Initial action
        action = np.zeros(7)
        # update total error for integral control
        self.eef_pos_overall_error += self.eef_to_peg_pos*self.dt
        moved_to_object = self.action_status['moved_to_object']
        grabbed = self.action_status['grabbed']
        moved_to_target = self.action_status['moved_to_target']
        done = self.action_status['done']
        

        if not moved_to_object:
            x_action = self.motion_P * self.eef_to_peg_pos #+ self.motion_I * self.eef_pos_overall_error 
            w_action = 0.02*np.array([0,0,self.eef_to_peg_quat[2],0])
            action = np.concatenate((x_action,w_action))      
            return action, self.action_status
        
        if moved_to_object and not grabbed:
            action = np.concatenate( (self.eef_to_peg_pos, np.array([0,0,0,1])))
            self.action_status['grabbed'] = True
            self.eef_pos_overall_error = 0 # renew the intergration to 0 for new conduction
            return action, self.action_status

        if grabbed and self.grab_wait_counter > 0:
            action = np.array([0,0,0,0,0,0,1])
            self.grab_wait_counter -= 1
            return action, self.action_status
           

        height = self.eef_pos[-1]
        if not moved_to_target and height<0.9:
            action = np.array([0, 0, 0.2, 0, 0, 0, 1])
            return action, self.action_status
        
        if not moved_to_target:
            action = np.concatenate( (self.motion_P * self.eef_to_hole_pos[0:2], np.array([0,0,0,0,1]) ) )
            return action, self.action_status
        
        # print("hole_to_peg_pos:\n \t x: {a[0]:2.4f}, y: {a[1]:2.4f}, z: {a[2]:2.4f}".format(a = self.hole_to_peg_pos))
        if moved_to_target and not done:
            if self.hole_to_peg_pos[2] < 0.035: # 0.05
                action = np.array([0, 0, 0, 0, 0, 0, -1])
                self.action_status['done'] = True
                return action, self.action_status

            action = np.array([0, 0, -0.1, 0, 0, 0, 1])
            return action, self.action_status


        if done:
            if height > 0.9:
                action = np.array([0, 0, 0, 0, 0, 0, 0])
                return action, self.action_status
            else:
                action = np.array([0, 0, 0.05, 0, 0, 0, 0])
                return action, self.action_status

