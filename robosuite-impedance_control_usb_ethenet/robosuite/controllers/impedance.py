from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
# from robosuite.controllers.impedance_equation import ImpedanceEq
import numpy as np
import math
from matplotlib import pyplot as plt
import numpy as np
# from robosuite.controllers.Impedance_param import kcm_impedance
from scipy.linalg import expm
from scipy.signal import savgol_filter
from copy import deepcopy

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}


class ImpedancePositionBaseController(Controller):
    """
    Controller for controlling robot arm via operational space control. Allows position and / or orientation control
    of the robot's end effector. For detailed information as to the mathematical foundation for this controller, please
    reference **add***

    NOTE: Control input actions can either be taken to be relative to the current position / orientation of the
    end effector or absolute values. In either case, a given action to this controller is assumed to be of the form:
    (x, y, z, ax, ay, az) if controlling pos and ori or simply (x, y, z) if only controlling pos

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        kp (float or Iterable of float): positional gain for determining desired torques based upon the pos / ori error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)

        impedance_mode (str): Impedance mode with which to run this controller. Options are {"fixed", "variable",
            "variable_kp"}. If "fixed", the controller will have fixed kp and damping_ratio values as specified by the
            @kp and @damping_ratio arguments. If "variable", both kp and damping_ratio will now be part of the
            controller action space, resulting in a total action space of (6 or 3) + 6 * 2. If "variable_kp", only kp
            will become variable, with damping_ratio fixed at 1 (critically damped). The resulting action space will
            then be (6 or 3) + 6.

        kp_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is set to either
            "variable" or "variable_kp". This sets the corresponding min / max ranges of the controller action space
            for the varying kp values. Can be either be a 2-list (same min / max for all kp action dims), or a 2-list
            of list (specific min / max for each kp dim)

        damping_ratio_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is
            set to "variable". This sets the corresponding min / max ranges of the controller action space for the
            varying damping_ratio values. Can be either be a 2-list (same min / max for all damping_ratio action dims),
            or a 2-list of list (specific min / max for each damping_ratio dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        position_limits (2-list of float or 2-list of Iterable of floats): Limits (m) below and above which the
            magnitude of a calculated goal eef position will be clipped. Can be either be a 2-list (same min/max value
            for all cartesian dims), or a 2-list of list (specific min/max values for each dim)

        orientation_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the
            magnitude of a calculated goal eef orientation will be clipped. Can be either be a 2-list
            (same min/max value for all joint dims), or a 2-list of list (specific min/mx values for each dim)

        interpolator_pos (Interpolator): Interpolator object to be used for interpolating from the current position to
            the goal position during each timestep between inputted actions

        interpolator_ori (Interpolator): Interpolator object to be used for interpolating from the current orientation
            to the goal orientation during each timestep between inputted actions

        control_ori (bool): Whether inputted actions will control both pos and ori or exclusively pos

        uncouple_pos_ori (bool): Whether to decouple torques meant to control pos and torques meant to control ori

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                 output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
                 kp=150,
                 damping_ratio=1,
                 impedance_mode="fixed",
                 kp_limits=(0, 300),
                 damping_ratio_limits=(0, 100),
                 policy_freq=20,
                 position_limits=None,
                 orientation_limits=None,
                 interpolator_pos=None,
                 interpolator_ori=None,
                 control_ori=True,
                 control_delta=True,
                 uncouple_pos_ori=True,
                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )

        # for ploting:
        self.plotter = True
        self.PartialImpedance = False
        # Determine whether this is pos ori or just pos
        self.use_ori = control_ori

        # Determine whether we want to use delta or absolute values as inputs
        self.use_delta = control_delta

        # Control dimension
        self.control_dim = 36 if self.use_ori else 3
        # self.name_suffix = "POSE" if self.use_ori else "POSITION"

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # kp kd
        self.kp = self.nums2array(kp, 6)
        # self.kp[1] *= 4
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio
        # self.kd[1] *= 3

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, "Error: Tried to instantiate IM_PB controller for unsupported " \
                                                  "impedance mode! Inputted impedance mode: {}, Supported modes: {}". \
            format(impedance_mode, IMPEDANCE_MODES)

        # Impedance mode
        self.impedance_mode = impedance_mode

        # Add to control dim based on impedance_mode
        if self.impedance_mode == "variable":
            self.control_dim += 36
        elif self.impedance_mode == "variable_kp":
            self.control_dim += 72

        # limits
        self.position_limits = position_limits
        self.orientation_limits = orientation_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # whether or not pos and ori want to be uncoupled
        self.uncoupling = uncouple_pos_ori

        # initialize goals based on initial pos / ori
        #   TODO: set goal to the final pos+ori
        self.goal_ori = np.array(self.initial_ee_ori_mat)#self.sim.data.get_body_xmat('peg1') #
        self.goal_pos = np.array(self.initial_ee_pos)

        self.relative_ori = np.zeros(3)
        self.ori_ref = None
        self.set_desired_goal = False
        self.desired_pos = np.zeros(12)
        self.torques = np.zeros(6)
        self.F0 = np.zeros(6)
        self.F_int = np.zeros(6)
        self.bias = 0
        # ee resets - bias at initial state
        self.ee_sensor_bias = deepcopy(self.sim.data.sensordata)

        # for graphs
        self.ee_pos_vec_x = []
        self.ee_pos_vec_y = []
        self.ee_pos_vec_z = []
        self.impedance_model_pos_vec_x = []
        self.impedance_model_pos_vec_y = []
        self.impedance_model_pos_vec_z = []
        self.error_pos_vec = []
        self.ee_ori_vec = []
        self.impedance_ori_vec = []
        self.wernce_vec_0 = []
        self.wernce_vec_int_Fx,self.wernce_vec_int_Fy,self.wernce_vec_int_Fz,self.wernce_vec_int_Mx,self.wernce_vec_int_My,self.wernce_vec_int_Mz = [],[],[],[],[],[]
        self.pos_min_jerk_x, self.pos_min_jerk_y, self.pos_min_jerk_z = [], [], []

    def set_goal(self, action, set_pos=None, set_ori=None):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_pos (Iterable): If set, overrides @action and sets the desired absolute eef position goal state
            set_ori (Iterable): IF set, overrides @action and sets the desired absolute eef orientation goal state
        """
        # TODO - set_goal(self, action, set_pos=None, set_ori=None) - possible to have problame with action dim
        # TODO - set_goal(self, action, set_pos=None, set_ori=None) - maybe this is here "set_desired_point" needed to be implamented.
        # Update state
        self.update()
        # Parse action based on the impedance mode, and update kp / kd as necessary
        if self.impedance_mode == "variable":
            damping_ratio, kp, delta = action[:6], action[6:12], action[12:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp) * np.clip(damping_ratio, self.damping_ratio_min, self.damping_ratio_max)
        elif self.impedance_mode == "variable_kp":
            kp, delta = action[:6], action[6:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp)  # critically damped
        else:  # This is case "fixed"
            delta = action

        self.goal_ori = set_ori
        self.goal_pos = set_pos

        # self.goal_pos = set_goal_position(scaled_delta[:3],
        #                                   np.array(self.sim.data.get_body_xpos("peg2"))+np.array([0, 0, 0.1]),
        #                                   position_limit=self.position_limits,
        #                                   set_pos=set_pos)

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref))  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

        #   set the desire_vec=goal
        self.set_desired_goal = True

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint.

        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stFalseanford.edu/publications/pdfs/Khatib_1987_RA.pdf

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()
        if self.set_desired_goal:
            self.desired_pos[:6] = np.concatenate((self.goal_pos, self.goal_ori), axis=0)
            self.desired_pos[6:12] = np.zeros(6)

        if self.bias < 850:
            self.ee_sensor_bias = deepcopy(self.sim.data.sensordata)
            self.bias += 1
        else:
            s=1
        self.F_int = (self.sim.data.sensordata-0*self.ee_sensor_bias)
        # self.F_int = np.concatenate((self.sim.data.cfrc_ext[21][3:],self.sim.data.cfrc_ext[21][:3]), axis=0)
        self.desired_pos = self.ImpedanceEq(self.ee_pos, T.mat2euler(self.ee_ori_mat), self.ee_pos_vel, self.ee_ori_vel,
                                           self.F_int, self.F0, self.desired_pos[:6], self.desired_pos[6:12],
                                       self.sim.model.opt.timestep)
        # print(self.sim.data.cfrc_ext[21])

        # ori_error = orientation_error(T.quat2mat(T.axisangle2quat(self.desired_pos[3:6].T)), self.ee_ori_mat)
        ori_error = orientation_error(T.euler2mat(self.desired_pos[3:6].T), self.ee_ori_mat)

        #       TODO - run_controller(self) - check if its OK (and vel_pos_error)
        # vel_ori_error = orientation_error(T.euler2mat(self.desired_pos[9:12].T), T.euler2mat(self.ee_ori_vel))
        # vel_ori_error = self.desired_pos[9:12] - self.ee_ori_vel
        vel_ori_error = - self.ee_ori_vel

        # Compute desired force and torque based on errors
        position_error = self.desired_pos[:3].T - self.ee_pos
        vel_pos_error = - self.ee_pos_vel

        if self.PartialImpedance:
            position_error = np.dot(np.linalg.pinv(self.K), self.F_int + self.F0)

        if self.plotter == True:
            # for checking:
            if (abs(position_error[0]) < 0.002 and abs(position_error[1]) < 0.002 and self.ee_pos[2] < 0.85) \
                    or self.model_timestep==998:
                        s=1
                        # self.control_plotter()

            # for graphs:
            self.ee_pos_vec_x.append(self.ee_pos[0])
            self.ee_pos_vec_y.append(self.ee_pos[1])
            self.ee_pos_vec_z.append(self.ee_pos[2])
            self.impedance_model_pos_vec_x.append(self.desired_pos[0])
            self.impedance_model_pos_vec_y.append(self.desired_pos[1])
            self.impedance_model_pos_vec_z.append(self.desired_pos[2])
            self.error_pos_vec.append(position_error)
            self.ee_ori_vec.append(T.mat2euler(self.ee_ori_mat))
            self.impedance_ori_vec.append(self.desired_pos[3:6])
            self.wernce_vec_0.append(self.torques)
            self.wernce_vec_int_Fx.append(self.F_int[0])
            self.wernce_vec_int_Fy.append(self.F_int[1])
            self.wernce_vec_int_Fz.append(self.F_int[2])
            self.wernce_vec_int_Mx.append(self.F_int[3])
            self.wernce_vec_int_My.append(self.F_int[4])
            self.wernce_vec_int_Mz.append(self.F_int[5])
            self.pos_min_jerk_x.append(self.goal_pos[0])
            self.pos_min_jerk_y.append(self.goal_pos[1])
            self.pos_min_jerk_z.append(self.goal_pos[2])
            # print(self.ee_sensor_bias)
            # print(self.sim.data.cfrc_ext)
            #print(self.ee_pos_vec)

        # F_r = kp * pos_err + kd * vel_err
        # self.kd[4:6] = np.zeros(2)
        # self.kp[4:6] = np.zeros(2)

        #################    calculate PD controller:         #########################################

        desired_force = (np.multiply(np.array(position_error), np.array(self.kp[0:3]))
                         + np.multiply(vel_pos_error, self.kd[0:3]))


        # Tau_r = kp * ori_err + kd * vel_err
        desired_torque = (np.multiply(np.array(ori_error), np.array(self.kp[3:6]))
                          + np.multiply(vel_ori_error, self.kd[3:6]))

        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(self.mass_matrix,
                                                                                 self.J_full,
                                                                                 self.J_pos,
                                                                                 self.J_ori)

        # Decouples desired positional control from orientation control
        if self.uncoupling:
            decoupled_force = np.dot(lambda_pos, desired_force.T)
            decoupled_torque = np.dot(lambda_ori, desired_torque.T)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)

        # Gamma (without null torques) = J^T * F + gravity compensations
        self.torques = np.dot(self.J_full.T, decoupled_wrench).reshape(6,) + self.torque_compensation

        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        self.torques += nullspace_torques(self.mass_matrix, nullspace_matrix,
                                          self.initial_joint, self.joint_pos, self.joint_vel)

        self.set_desired_goal = False
        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def reset_goal(self):
        """
        Resets the goal to the current state of the robot
        """
        self.goal_ori = np.zeros(3)
        self.goal_pos = np.array(self.sim.data.get_body_xpos("peg1"))

        # Also reset interpolators if required

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref))  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.impedance_mode == "variable":
            low = np.concatenate([self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate([self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max
        return low, high

    @property
    def name(self):
        return 'IMPEDANCE_PB'

    def ImpedanceEq(self,x0, th0, x0_d, th0_d, F_int, F0, xm_pose, xmd_pose, dt):
        """
        Impedance Eq: F_int-F0=K(x0-xm)+C(x0_d-xm_d)-Mxm_dd

        Solving the impedance equation for x(k+1)=Ax(k)+Bu(k) where
        x(k+1)=[Xm,thm,Xm_d,thm_d]

        Parameters:
            x0,x0_d,th0,th0_d - desired goal position/orientation and velocity
            F_int - measured force/moments in [N/Nm] (what the robot sense)
            F0 - desired applied force/moments (what the robot does)
            xm_pose - impedance model (updated in a loop) initialized at the initial pose of robot
            A_d, B_d - A and B matrices of x(k+1)=Ax(k)+Bu(k)
        Output:
            X_nex = x(k+1) = [Xm,thm,Xm_d,thm_d]
        """

        # state space formulation
        # X=[xm;thm;xm_d;thm_d] U=[F_int;M_int;x0;th0;x0d;th0d]
        A_1 = np.concatenate((np.zeros([6, 6], dtype=int), np.identity(6)), axis=1)
        A_2 = np.concatenate((np.dot(-np.linalg.pinv(self.M), self.K), np.dot(-np.linalg.pinv(self.M), self.C)), axis=1)
        A_temp = np.concatenate((A_1, A_2), axis=0)

        B_1 = np.zeros([6, 18], dtype=int)
        B_2 = np.concatenate((np.linalg.pinv(self.M), np.dot(np.linalg.pinv(self.M), self.K),
                              np.dot(np.linalg.pinv(self.M), self.C)), axis=1)
        B_temp = np.concatenate((B_1, B_2), axis=0)

        # discrete state space A, B matrices
        A_d = expm(A_temp * dt)
        B_d = np.dot(np.dot(np.linalg.pinv(A_temp), (A_d - np.identity(A_d.shape[0]))), B_temp)

        # defining goal vector of position/ velocity inside the hole
        X0 = np.concatenate((x0, th0), axis=0).reshape(6, 1)  # const
        X0d = np.concatenate((x0_d, th0_d), axis=0).reshape(6, 1)  # const

        # impedance model xm is initialized to initial position of the EEF and modified by force feedback
        xm = xm_pose[:3].reshape(3, 1)
        thm = xm_pose[3:].reshape(3, 1)
        xm_d = xmd_pose[:3].reshape(3, 1)
        thm_d = xmd_pose[3:].reshape(3, 1)

        # State Space vectors
        X = np.concatenate((xm, thm, xm_d, thm_d), axis=0)  # 12x1 column vector
        zero_arr = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        # U = np.concatenate((zero_arr, zero_arr, x0, th0, zero_arr, zero_arr), axis=0)
        U = np.concatenate((F0 - F_int, x0, th0, x0_d, th0_d), axis=0).reshape(18, 1)

        # discrete state solution X(k+1)=Ad*X(k)+Bd*U(k)
        X_nex = np.dot(A_d, X) + np.dot(B_d, U)
        # X_nex = np.round(X_nex, 10)

        return X_nex.reshape(12,)

    def set_control_param(self, action):
        # self.learn += status
        #   TODO - set_control_param(self, action) - add cases of different learning- just K; K+C ect.
        # if self.learn == 1:
        self.K = action.reshape(6,6) #*100
            # self.ee_sensor_bias = deepcopy(self.sim.data.sensordata)
        # print(self.K)
            # self.learn =+ 1
        # from numpy import linalg as LA
        # w, v = LA.eig(self.K)
        # print(self.K)
        # if self.learn == 0:
        K =np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0,0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
            # self.ee_sensor_bias = deepcopy(self.sim.data.sensordata)

        K =np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0,0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

        K = np.array([[3.9805126, 4.322858, 3.1552098, 0., 0., 0.],
                     [5.3451514 ,0., 5.061681, 0., 2.4784195, 0.],
                     [2.2692046, 3.4668186, 0., 0., 0.,  0.],
                     [0., 0., 2.9420927, 0., 0., 1.4223151],
                     [0., 1.7808434, 0., 5.5859084, 4.6935296, 0.],
                     [0., 7.637603, 0.7636627, 0.,  0.,  1.8498616]])

        K = np.array([[3.416321, 4.0116177, 3.2101188, 0., 0., 0.],
                     [5.471027, 0., 4.8644342, 0., 2.4466724, 0.],
                    [2.3800824, 3.5235841, 0., 0., 0., 0.],
                    [0., 0., 3.486249, 0., 0., 1.5501156],
                    [0., 1.5652821, 0., 5.3258753, 4.3522663, 0.],
                    [0., 7.511775, 0.6735706, 0., 0., 1.5767453]])

        g_pos = 8.51427155e+01 * 10
        g_orient = 2.58588562e+00 * 80
        m = 0.2

        self.M = self.mass_matrix
        # self.M = np.identity(6) * m



        #C = np.zeros([6,6])
        C = np.array([[g_pos, 0, 0, 0, 0, 0],
                      [0, g_pos, 0, 0, 0, 0],
                      [0, 0, g_pos, 0, 0, 0],
                      [0, 0, 0, g_orient, 0, 0],
                      [0, 0, 0, 0, g_orient, 0],
                      [0, 0, 0, 0, 0, g_orient]])
        # if self.learn == 1:
        K = np.array([[g_pos, 0, 0, 0, 0, 0],
                      [0, g_pos, 0, 0, 0, 0],
                      [0, 0, g_pos, 0, 0, 0],
                      [0, 0, 0, g_orient, 0, 0],
                      [0, 0, 0, 0, g_orient, 0],
                      [0, 0, 0, 0, 0, g_orient]])

        # critical damping C = 2*sqrt(KM)
        # self.C = np.nan_to_num(2 * np.sqrt(np.dot(self.K, self.M)))
        self.C = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]])
    def control_plotter(self):

        pass
        t = list(range(0, np.size(self.ee_pos_vec_x)))
        plt.figure()
        plt.plot(t, savgol_filter(self.impedance_model_pos_vec_x, 601, 4), 'b--', label='Xm position')
        plt.plot(t, self.ee_pos_vec_x, 'b', label='Xr position')
        plt.plot(t, self.pos_min_jerk_x, 'r--', label='X_ref position')
        plt.legend()
        plt.ylabel('Position [m]')
        plt.show()

        t = list(range(0, np.size(self.ee_pos_vec_x)))
        plt.figure()
        plt.plot(t, savgol_filter(self.impedance_model_pos_vec_y, 601, 4), 'g--', label='Ym position')
        plt.plot(t, self.ee_pos_vec_y, 'g', label='Yr position')
        plt.plot(t, self.pos_min_jerk_y, 'r--', label='Y_ref position')
        plt.legend()
        plt.ylabel('Position [m]')
        plt.show()

        t = list(range(0, np.size(self.ee_pos_vec_x)))
        plt.figure()
        plt.plot(t, savgol_filter(self.impedance_model_pos_vec_z, 601, 4), 'g--', label='Zm position')
        plt.plot(t, self.ee_pos_vec_z, 'g', label='Zr position')
        plt.plot(t, self.pos_min_jerk_z, 'r--', label='Z_ref position')
        plt.legend()
        plt.ylabel('Position [m]')
        plt.show()

        t = list(range(0, np.size(self.ee_pos_vec_x)))
        plt.figure()
        plt.plot(t, self.impedance_ori_vec, '--')
        plt.plot(t, self.ee_ori_vec)
        plt.legend(['X_ori_m', 'Y_ori_m', 'Z_ori_m', 'X_ori_r', 'Y_ori_r', 'Z_ori_r'])
        plt.ylabel('ori [rad]')
        plt.show()

        t = list(range(0, np.size(self.ee_pos_vec_x)))
        plt.figure()
        plt.plot(t, self.error_pos_vec)
        plt.legend(['X error', 'Y error', 'Z error'])
        plt.ylabel('Position Error [m]')
        plt.show()

        t = list(range(0, np.size(self.ee_pos_vec_x)))
        plt.figure()
        plt.plot(t, self.wernce_vec_0)
        plt.legend(['Fx', 'Fy', 'Fz','Mx', 'My', 'Mz'])
        plt.ylabel('wernce [N or Nm]')
        plt.show()

        t = list(range(0, np.size(self.ee_pos_vec_x)))
        plt.figure()
        plt.plot(t, savgol_filter(self.wernce_vec_int_Fx, 601, 9), label='Fx_int')
        plt.plot(t, savgol_filter(self.wernce_vec_int_Fy, 601, 9), label='Fy_int')
        plt.plot(t, savgol_filter(self.wernce_vec_int_Fz, 601, 9), label='Fz_int')
        plt.legend(['Fx_int', 'Fy_int', 'Fz_int'])
        plt.ylabel('Force [N]')
        plt.show()
        #
        t = list(range(0, np.size(self.ee_pos_vec_x)))
        plt.figure()
        plt.plot(t, savgol_filter(self.wernce_vec_int_Mx, 601, 9), label='Fx_int')
        plt.plot(t, savgol_filter(self.wernce_vec_int_My, 601, 9), label='Fy_int')
        plt.plot(t, savgol_filter(self.wernce_vec_int_Mz, 601, 9), label='Fz_int')
        plt.legend(['Mx_int','My_int','Mz_int'])
        plt.ylabel('Torque [Nm]')
        plt.show()