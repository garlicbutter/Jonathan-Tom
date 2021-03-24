import numpy as np

from collections import OrderedDict
from mujoco_py import MjSim, MjRenderContextOffscreen
from mujoco_py import load_model_from_xml
import robosuite.utils.transform_utils as T

from robosuite.utils import SimulationError, XMLError, MujocoPyRenderer

REGISTERED_ENVS = {}


def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


def make(env_name, *args, **kwargs):
    """
    Instantiates a robosuite environment.

    This method attempts to mirror the equivalent functionality of gym.make in a somewhat sloppy way.

    Args:
        env_name (str): Name of the robosuite environment to initialize
        *args: Additional arguments to pass to the specific environment class initializer
        **kwargs: Additional arguments to pass to the specific environment class initializer

    Returns:
        MujocoEnv: Desired robosuite environment

    Raises:
        Exception: [Invalid environment name]
    """
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )
    return REGISTERED_ENVS[env_name](*args, **kwargs)


class EnvMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["MujocoEnv", "RobotEnv"]

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls


class MujocoEnv(metaclass=EnvMeta):
    """
    Initializes a Mujoco Environment.

    Args:
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering.

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes
            in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes
            in camera. False otherwise.

        control_freq (float): how many control signals to receive
            in every simulated second. This sets the amount of simulation time
            that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

    Raises:
        ValueError: [Invalid renderer selection]
    """

    def __init__(
        self,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        action_space_def="Impedance_K"
    ):
        # First, verify that both the on- and off-screen renderers are not being used simultaneously
        if has_renderer is True and has_offscreen_renderer is True:
            raise ValueError("the onscreen and offscreen renderers cannot be used simultaneously.")

        # define learning space - and number of learning parameters (action space)
        self.action_space_def=action_space_def

        # Rendering-specific attributes
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.render_camera = render_camera
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.viewer = None

        # Simulation-specific attributes
        self.control_freq = control_freq
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.hard_reset = hard_reset
        self.model = None
        self.cur_time = None
        self.model_timestep = None
        self.control_timestep = None
        self.deterministic_reset = False            # Whether to add randomized resetting of objects / robot joints
        self.loop_time = ((1 / self.control_freq) * self.horizon)
        self.trans = {'0':0.2}# , '1':0.005} #{'0':0.2 , '1':0.05}
        self.all_time = 0
        self.rewards_all = []

        # for checkink graph of min jerk:
        self.des_pos = []

        self.peg_pos = np.zeros(3)
        self.hole_pos = [np.inf,np.inf,np.inf]
        # Load the model
        self._load_model()

        # Initialize the simulation
        self._initialize_sim()

        # Run all further internal (re-)initialization required
        self._reset_internal()

    def initialize_time(self, control_freq):
        """
        Initializes the time constants used for simulation.

        Args:
            control_freq (float): Hz rate to run control loop at within the simulation
        """
        self.cur_time = 0
        self.model_timestep = self.sim.model.opt.timestep
        if self.model_timestep <= 0:
            raise XMLError("xml model defined non-positive time step")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError(
                "control frequency {} is invalid".format(control_freq)
            )
        self.control_timestep = 1. / control_freq

    def _load_model(self):
        """Loads an xml model, puts it in self.model"""
        pass

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        pass

    def _initialize_sim(self, xml_string=None):
        """
        Creates a MjSim object and stores it in self.sim. If @xml_string is specified, the MjSim object will be created
        from the specified xml_string. Else, it will pull from self.model to instantiate the simulation

        Args:
            xml_string (str): If specified, creates MjSim object from this filepath
        """
        # if we have an xml string, use that to create the sim. Otherwise, use the local model
        self.mjpy_model = load_model_from_xml(xml_string) if xml_string else self.model.get_model(mode="mujoco_py")

        # Create the simulation instance and run a single step to make sure changes have propagated through sim state
        self.sim = MjSim(self.mjpy_model)
        self.sim.step()

        # Setup sim time based on control frequency
        self.initialize_time(self.control_freq)

    def reset(self):
        """
        Resets simulation.

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # Use hard reset if requested
        if self.hard_reset and not self.deterministic_reset:
            self._destroy_viewer()
            self._load_model()
            self._initialize_sim()
        # Else, we only reset the sim internally
        else:
            self.sim.reset()
        # Reset necessary robosuite-centric variables
        self._reset_internal()
        self.sim.forward()
        return self._get_observation()

    def _reset_internal(self):
        """Resets simulation internal configurations."""

        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (1 if self.render_collision_mesh else 0)
            self.viewer.viewer.vopt.geomgroup[1] = (1 if self.render_visual_mesh else 0)

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

            # make sure mujoco-py doesn't block rendering frames
            # (see https://github.com/StanfordVL/robosuite/issues/39)
            self.viewer.viewer._render_every_frame = True

            # Set the camera angle for viewing
            if self.render_camera is not None:
                self.viewer.set_camera(camera_id=self.sim.model.camera_name2id(self.render_camera))

        elif self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(self.sim)
                self.sim.add_render_context(render_context)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = (1 if self.render_collision_mesh else 0)
            self.sim._render_context_offscreen.vopt.geomgroup[1] = (1 if self.render_visual_mesh else 0)

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0
        self.rewards_all= []
        self.done = False
        self.des_pos = []

    def _get_observation(self):
        """
        Grabs observations from the environment.

        Returns:
            OrderedDict: OrderedDict containing observations [(name_string, np.array), ...]

        """
        return OrderedDict()

    def step(self, action):
        """
        Takes a step in simulation with control command @action.

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information

        Raises:
            ValueError: [Steps past episode termination]

        """
        #   TODO - step(self, action) - add lines to this function (like base orens)
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.all_time += 1

        done = False
        # # if self.use_camera_obs:
        # self.timestep += 1
        # policy_step = True
        #
        # for i in range(int(self.control_timestep / self.model_timestep)):
        #     self.sim.forward()
        #     self._pre_action(action, policy_step)
        #     self.sim.step()
        #     policy_step = False
        #
        # # if self.has_renderer:
        # #      self.render()
        # # Note: this is done all at once to avoid floating point inaccuracies
        # self.cur_time += self.control_timestep
        #
        # reward, done, info = self._post_action(action)
        #
        # info.update({"time": self.timestep})
        # # print(np.sum(rewards_all))
        # if self.success > 1:
        #     info.update({"is_success": True})
        # else:
        #     info.update({"is_success": False})
        # if done and self.success > 1:
        #     print("----------------:)------------------")
        #     reward =+ 100000
        # elif done and sum(self.rewards_all) < 800:
        #     reward = -600
        # self.rewards_all.append(reward)
        # return self._get_observation(), reward, done, info
        # else:
        while not done:
            self.timestep += 1
            '''
                Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
                multiple torque commands in between new high level action commands. Therefore, we need to denote via
                'policy_step' whether the current step we're taking is simply an internal update of the controller,
                or an actual policy update
            '''

            policy_step = True
            if self.has_renderer:
                 self.render()
            # self._dict_action(action)
            '''
                Loop through the simulation at the model timestep rate until we're ready to take the next policy step
                (as defined by the control frequency specified at the environment level)
            '''

            for i in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                self._pre_action(action, policy_step)
                self.sim.step()
                policy_step = False

            # Note: this is done all at once to avoid floating point inaccuracies
            self.cur_time += self.control_timestep

            reward, done, info = self._post_action(action)
            self.rewards_all.append(reward)

        info.update({"time": self.timestep})
        # print(np.sum(rewards_all))
        if self.success > 1:
            info.update({"is_success": True})
        else:
            info.update({"is_success": False})
        info.update({"episode": self.timestep})
        if done and self.success > 1:
            print("----------------:)------------------")
            # reward =+ 100000
            self.rewards_all = []
            self.rewards_all = 40000
            return self._get_observation(), np.sum(self.rewards_all) if np.sum(self.rewards_all) > 0 else 0, done, info
        elif done and sum(self.rewards_all) < 800:
            reward = -600
        self.rewards_all.append(reward)
        return self._get_observation(), np.sum(self.rewards_all) if np.sum(self.rewards_all) > 0 else 0, done, info

    def _pre_action(self, action, policy_step=False):
        """
        Do any preprocessing before taking an action.

        Args:
            action (np.array): Action to execute within the environment
            policy_step (bool): Whether this current loop is an actual policy step or internal sim update step
        """
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method

        """
        reward = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        self.done = ((self.timestep >= self.horizon) or (self._check_success() and self.checked ==
                                            (self.num_via_points-1) and self.success == 2) and not self.ignore_done)
        # if self.done:
            # print(reward)
        return reward, self.done, {}

    def reward(self, action):
        """
        Reward should be a function of state and action

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            float: Reward from environment
        """
        raise NotImplementedError

    def render(self):
        """
        Renders to an on-screen window.
        """
        self.viewer.render()

    def observation_spec(self):
        """
        Returns an observation as observation specification.

        An alternative design is to return an OrderedDict where the keys
        are the observation names and the values are the shapes of observations.
        We leave this alternative implementation commented out, as we find the
        current design is easier to use in practice.

        Returns:
            OrderedDict: Observations from the environment
        """
        observation = self._get_observation()
        return observation

    @property
    def action_spec(self):
        """
        Action specification should be implemented in subclasses.

        Action space is represented by a tuple of (low, high), which are two numpy
        vectors that specify the min/max action limits per dimension.
        """
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        Size of the action space
        Returns:
            int: Action space dimension
        """
        raise NotImplementedError

    def reset_from_xml_string(self, xml_string):
        """
        Reloads the environment from an XML description of the environment.

        Args:
            xml_string (str): Filepath to the xml file that will be loaded directly into the sim
        """

        # if there is an active viewer window, destroy it
        self.close()

        # Since we are reloading from an xml_string, we are deterministically resetting
        self.deterministic_reset = True

        # initialize sim from xml
        self._initialize_sim(xml_string=xml_string)

        # Now reset as normal
        self.reset()

        # Turn off deterministic reset
        self.deterministic_reset = False

    def find_contacts(self, geoms_1, geoms_2):
        """
        Finds contact between two geom groups.

        Args:
            geoms_1 (list of str): a list of geom names
            geoms_2 (list of str): another list of geom names

        Returns:
            generator: iterator of all contacts between @geoms_1 and @geoms_2
        """
        for contact in self.sim.data.contact[0 : self.sim.data.ncon]:
            # check contact geom in geoms
            c1_in_g1 = self.sim.model.geom_id2name(contact.geom1) in geoms_1
            c2_in_g2 = self.sim.model.geom_id2name(contact.geom2) in geoms_2
            # check contact geom in geoms (flipped)
            c2_in_g1 = self.sim.model.geom_id2name(contact.geom2) in geoms_1
            c1_in_g2 = self.sim.model.geom_id2name(contact.geom1) in geoms_2
            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
                yield contact

    def _check_success(self):
        """
        Checks if the task has been completed. Should be implemented by subclasses

        Returns:
            bool: True if the task has been completed
        """
        raise NotImplementedError

    def _destroy_viewer(self):
        """
        Destroys the current mujoco renderer instance if it exists
        """
        # if there is an active viewer window, destroy it
        if self.viewer is not None:
            self.viewer.close()  # change this to viewer.finish()?
            self.viewer = None

    def _calc_D(self, t0, tf):

        return np.array([
            [1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
            [0, 1, 2 * t0, 3 * (t0) ** 2, 4 * (t0) ** 3, 5 * (t0) ** 4],
            [0, 0, 2, 6 * (t0), 12 * (t0) ** 2, 20 * (t0) ** 3],
            [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
            [0, 1, 2 * tf, 3 * (tf) ** 2, 4 * (tf) ** 3, 5 * (tf) ** 4],
            [0, 0, 2, 6 * (tf), 12 * (tf) ** 2, 20 * (tf) ** 3]
        ])

    def _calc_a(self, S, t0, tf):
        D = self._calc_D(t0, tf)
        a = np.dot(np.linalg.inv(D), S)

        return a

    def calc_kinematics(self, t, a):

        D = self._calc_D(t, t)
        s = np.dot(D, a)

        return s

    def built_min_jerk_traj(self):
        '''
        built minimum jerk (the desired trajectory)

        return:
        '''
        pos = self.via_points['p' + str(self.checked)][:3]
        ori = self.via_points['p' + str(self.checked)][3:]

        # if self.checked == 0:
        #     pos_in = pos
        #     ori_in = ori
        # else:
        #     pos_in = self.via_points['p' + str(self.checked-1)][:3]
        #     ori_in = self.via_points['p' + str(self.checked-1)][3:]
        #     pos_in[1] = pos[1]
        #     pos_in[0] = pos[0]
        # if self.timestep < 10:
        #     pos_in = self.peg_pos
        pos_in = self.peg_pos
        ori_in = T.mat2euler(self.sim.data.get_body_xmat("robot0_right_hand"))

        if self.checked == 0 :
            pos_in = self.pos_in[0]
            ori_in = self.pos_in[1]
            pos_in[2] = 1.0
        if self.num_via_points > 2:
            if self.checked == 1 :
                pos_in = pos  # self.peg_pos
                ori_in = ori  # T.mat2euler(self.sim.data.get_body_xmat("robot0_right_hand"))

        self.X_fi = np.array([pos[0], 0, 0])
        self.Y_fi = np.array([pos[1], 0, 0])
        self.Z_fi = np.array([pos[2], 0, 0])

        self.X_t_fi = np.array([ori[0], 0, 0])
        self.Y_t_fi = np.array([ori[1], 0, 0])
        self.Z_t_fi = np.array([ori[2], 0, 0])

        X_in = np.array([pos_in[0], 0, 0])
        Y_in = np.array([pos_in[1], 0, 0])
        Z_in = np.array([pos_in[2], 0, 0])

        X_t_in = np.array([ori_in[0], 0, 0])
        Y_t_in = np.array([ori_in[1], 0, 0])
        Z_t_in = np.array([ori_in[2], 0, 0])

        if self.checked < (self.num_via_points-1):
            time_to = self.loop_time * (self.trans[str(self.checked)])
        else:
            time_to = self.loop_time * (1 - self.trans[str(self.checked-1)]) * 0.3
            # time_to = self.loop_time * (1 - sum(self.trans[str(x)] for x in range(self.num_via_points - 1)))  # * 0.1
        S1_loc = np.array([X_in[0], 0, 0, self.X_fi[0], 0, 0])
        self.a1_loc = self._calc_a(S1_loc, 0, time_to)

        self.loc_x_y = (self.Y_fi - Y_in) / (self.X_fi - X_in + 1e-6)
        self.loc_x_z = (self.Z_fi - Z_in) / (self.X_fi - X_in + 1e-6)
        self.b_y = (self.X_fi * Y_in - X_in * self.Y_fi) / (self.X_fi - X_in + 1e-6)
        self.b_z = (self.X_fi * Z_in - X_in * self.Z_fi) / (self.X_fi - X_in + 1e-6)
        s1_angle = np.array([ori[0], 0, 0, ori[0], 0, 0])
        self.a1_angle = self._calc_a(s1_angle, 0, time_to)

        self.teta_x_y = (self.Y_t_fi - Y_t_in) / (self.X_t_fi - X_t_in + 1e-6)
        self.teta_x_z = (self.Z_t_fi - Z_t_in) / (self.X_t_fi - X_t_in + 1e-6)

    def built_next_desired_point(self):
        t_now = self.sim.data.time % (self.loop_time / self.num_via_points)
        kinem_x = self.calc_kinematics(t_now, self.a1_loc)[:3]

        # kinem_y = self.loc_x_y * kinem_x  + self.b_y
        # kinem_z = self.loc_x_z * kinem_x +self.b_z
        kinem_y = self.loc_x_y * (kinem_x - self.X_fi) + self.Y_fi
        kinem_z = self.loc_x_z * (kinem_x - self.X_fi) + self.Z_fi

        kinem_teta_x = self.calc_kinematics(t_now, self.a1_angle)[:3]
        kinem_teta_y = self.teta_x_y * (kinem_teta_x - self.X_t_fi) + self.Y_t_fi
        kinem_teta_z = self.teta_x_z * (kinem_teta_x - self.X_t_fi) + self.Z_t_fi

        right_pos_desired = np.array([kinem_x[0], kinem_y[0], kinem_z[0]])
        right_ori_desired = np.array([kinem_teta_x[0], kinem_teta_y[0], kinem_teta_z[0]])

        right_vel_desired = np.array([kinem_x[1], kinem_y[1], kinem_z[1]])
        right_omega_desired = np.array([kinem_teta_x[1], kinem_teta_y[1], kinem_teta_z[1]])
        # for checking - graphs:
        self.des_pos.append(right_pos_desired)
        return [right_pos_desired, right_ori_desired, right_vel_desired, right_omega_desired]

    def calc_next_desired_point(self):
        """"
        create minimum jerk trajectory.
        calculate the next desired point in the desired trajectory.
        first, built minimum jerk trajectory, then, built the next desired point,
        taking in consideration hhe current time and calculate it.
        base on: https://www.researchgate.net/publication/329954197_A_Novel_Tuning_Method_for_PD_Control_of_Robotic_Manipulators_Based_on_Minimum_Jerk_Principle

        return: vec of desired pos, ori, vel and ori_dot (size 1,12)


        """
        if self.checked < (self.num_via_points-1):
            self.switch += self._switch_via_point()
            self.switch_seq += (self._switch_via_point() * self.timestep)
            # if self.switch == 3 or self.timestep == round((self.horizon * (sum(self.trans[str(x)] for x in range(self.checked+1))))):
            if self.switch == 3 or self.timestep == round((self.horizon * self.trans[str(self.checked)])):
                self.checked += 1
                self.built_min_jerk_traj()
            else:
                if self.switch > 3:
                    self.switch = 0
                    self.switch_seq = 0
            # if self.timestep == (self.horizon * (self.trans)):
            #     self.checked += 1
            #     if self.checked > 1:
            #         print(f"timestep = {self.timestep}\n{self.horizon * self.trans}")
            #     self.built_min_jerk_traj()

        if self.timestep == 1:
            self.built_min_jerk_traj()

        return self.built_next_desired_point()

    def close(self):
        """Do any cleanup necessary here."""
        self._destroy_viewer()
