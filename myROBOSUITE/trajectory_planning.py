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
        time_sections = 5,
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
        self.time_sections      =   time_sections
        self.dt_counter         =   0
        self.init               =   1
        self.t_section          =   0
        # Temporary put it here, can be changed as input in main to decide the action as delta or abs value 
        self.action_mode        =   "abs_position"
        # action status
        self.action_status      ={'moved_to_object':False,
                                'grabbed'		  :False,
                                'moved_to_target':False,}

    def update_obs(self,obs):
        # Update Observed value
        self.eef_pos 		=   obs['robot0_eef_pos'] 					# array 1x3
        self.eef_quat 		=   obs['robot0_eef_quat'] 					# array 1x4
        self.eef_quat[0]    =   self.eef_quat[1]
        self.eef_quat[1]    =   self.eef_quat[0]
        self.eef_quat[2]    =   -self.eef_quat[2]
        self.eef_vel        =   obs['robot0_gripper_qvel'][0:3]
        self.peg_pos 		=   obs['peg_pos'] + np.array([0,0,0.025])	# array 1x3
        self.peg_quat 		=   obs['peg_quat']							# array 1X4
        self.eef_to_peg_pos =   self.peg_pos - self.eef_pos							# array 1x3
        self.eef_to_peg_quat = self.peg_quat - self.eef_quat						# array 1x3
        self.hole_pos 		= obs['plate_pos'] + quat2mat(obs['plate_quat']) @ np.array([0.155636,0.1507,0.1])	# array 1x3
        self.eef_to_hole_pos = self.hole_pos - self.eef_pos						# array 1x3
        # decide status based on the observation
        record = self.action_status
        self.decide_status()
        if record != self.action_status or self.init:
            self.change_inital_conditions()
            self.init = 0

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
            self.dt_counter = 0

        if self.action_status['moved_to_object'] and not self.action_status['grabbed']:
            self.action_status['grabbed'] = True
            self.dt_counter = 0

        if self.action_status['grabbed'] and norm(self.eef_to_hole_pos[0:2]) < 0.005:
            self.action_status['moved_to_target'] = True
            self.dt_counter = 0

        return self.action_status

    def get_policy_action(self):
        
        '''
        take obs as inputs
        return motion as output
        '''

        # Initial action
        action = np.zeros(7)

        '''
        Two different modes can be chosed:
        self.action_mode == 'delta value': The action is given as delta value in @action, close to velocity control
        self.action_mode == 'abs_position': The action is given as absolute position value in world frame, close to position control
        '''

        if self.action_mode == 'delta value':
            # update total error for integral control
            self.eef_pos_overall_error += self.eef_to_peg_pos*self.dt

            if not self.action_status['moved_to_object']:
                x_action = self.motion_P * self.eef_to_peg_pos + self.motion_I * self.eef_pos_overall_error 
                w_action = 0.01*np.array([0,0,self.eef_to_peg_quat[2],0])
                action = np.concatenate((x_action,w_action))      
                return action
            
            if self.action_status['moved_to_object']:
                action = np.concatenate( (self.eef_to_peg_pos, np.array([0,0,0,1]))) 
                self.eef_pos_overall_error = 0 # renew the intergration to 0 for new conduction
                return action

            if self.action_status['grabbed']:
                if not self.action_status['raised']:
                    action = np.array([0, 0, 0.2, 0, 0, 0, 1])
                    return action

                if self.action_status['raised']:
                    self.eef_pos_overall_error = 0 # renew the intergration to 0 for new conduction
                    return action
                
                if not self.action_status['moved_to_target']:
                    action = np.concatenate( (self.eef_to_hole_pos[0:2], np.array([0,0,0,0,1]) ) )
                    return action
                
                if self.action_status['moved_to_target']:
                    action = np.array([0, 0, -0.1, 0, 0, 0, 1])
                    return action

        if self.action_mode == 'abs_position':

            T = self.time_sections
            q_i = self.section_pos_init
            q_f = self.section_pos_final
            a0 = q_i
            a3 = 20*(q_f-q_i)/(2*T**3)
            a4 = 30*(q_i-q_f)/(2*T**4)
            a5 = 12*(q_f-q_i)/(2*T**5)
            action = a0 + a3*self.t_section**3 + a4*self.t_section**4 + a5*self.t_section**5
            self.dt_counter = self.dt_counter+1
            self.t_section = self.dt_counter * self.dt

            return action

    def change_inital_conditions(self):

        if not self.action_status['moved_to_object']:
            self.section_pos_init = np.concatenate((self.eef_pos,[0,0,0,0]))
            self.section_pos_final = np.concatenate((self.peg_pos,[0,0,0,0]))

        if self.action_status['moved_to_object']:
            self.section_pos_init = np.concatenate((self.eef_pos,[0,0,0,1]))
            self.section_pos_final = np.concatenate((self.hole_pos,[0,0,0,1]))

        if self.action_status['grabbed']:
            if not self.action_status['raised']:
                self.section_pos_init = np.concatenate((self.eef_pos,[0,0,0,1]))
                self.section_pos_final = self.eef_pos + np.array([0, 0, 1, 0, 0, 0, 1])

            if (self.action_status['raised']) and (not self.action_status['moved_to_target']):
                self.section_pos_init = np.concatenate((self.eef_pos,[0,0,0,1]))
                self.section_pos_final = np.concatenate( (self.hole_pos[0:2], np.array([0,0,0,0,1]) ) )

            if self.action_status['moved_to_target']:
                self.section_pos_init = np.concatenate((self.eef_pos,[0,0,0,1]))
                self.section_pos_final = np.array([0, 0, -1, 0, 0, 0, 1])
