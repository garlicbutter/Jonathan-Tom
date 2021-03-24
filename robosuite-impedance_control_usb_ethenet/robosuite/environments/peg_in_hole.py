from collections import OrderedDict
import random
import numpy as np
from copy import deepcopy

import robosuite.utils.transform_utils as T
from robosuite.environments.robot_env import RobotEnv
from robosuite.models import MujocoWorldBase
from robosuite.robots import SingleArm
from robosuite.controllers import controller_factory, load_controller_config
from robosuite.models.arenas import TableArena, EmptyArena
from robosuite.models.objects import PlateWith2_5mmHoleObject, PlateWith5mmHoleObject, CylinderObject, PlateWithHoleObject
from robosuite.models.tasks import ManipulationTask, ManipulationTaskNoGripperFunc, SequentialCompositeSampler
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string


class PegInHole(RobotEnv):
    """
    This class corresponds to the nut assembly task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        gripper_visualizations (bool or list of bool): True if using gripper visualization.
            Useful for teleoperation. Should either be single bool if gripper visualization is to be used for all
            robots or else it should be a list of the same length as "robots" param

        placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler instance): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        single_object_mode (int): specifies which version of the task to do. Note that
            the observations change accordingly.

            :`0`: corresponds to the full task with both types of nuts.

            :`1`: corresponds to an easier task with only one type of nut initialized
               on the table with every reset. The type is randomized on every reset.

            :`2`: corresponds to an easier task with only one type of nut initialized
               on the table with every reset. The type is kept constant and will not
               change between resets.

        nut_type (string): if provided, should be either "round" or "square". Determines
            which type of nut (round or square) will be spawned on every environment
            reset. Only used if @single_object_mode is 2.

        use_indicator_object (bool): if True, sets up an indicator object that
            is useful for debugging.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid nut type specified]
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        controller_configs=None,
        gripper_types="default",
        gripper_visualizations=False,
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        peg_radius=(0.00125, 0.00125), #(0.00125, 0.00125)
        peg_length=0.08,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_robotview",#('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand').
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        dist_error=0.0015,
        angle_error=0,
        checked=0,
        switch_seq=0,
        switch=0,
        success=0,
        placement_initializer=None,
    ):
        # First, verify that only one robot is being inputted
        self._check_robot_configuration(robots)

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            # treat sampling of each type of nut differently since we require different
            # sampling ranges for each
            self.placement_initializer = SequentialCompositeSampler()
            self.placement_initializer.sample_on_top(
                "hole",
                surface_name="table",
                x_range=[-0.1, 0.01],#[-0.115, -0.11],[-0.3, -0.2]
                y_range=[-0.11, 0.01],#[0.11, 0.225],
                rotation=None,
                rotation_axis='z',
                z_offset=0.02,
                ensure_object_boundary_in_range=False,
            )
        # set learning dict
        self.action_dict = OrderedDict()

        # Save peg specs
        self.peg_radius = peg_radius
        self.peg_length = peg_length

        self.dist_error = dist_error
        self.angle_error = angle_error
        self.num_via_points = 2
        self.first_via_points = OrderedDict()
        self.via_points = OrderedDict()
        self.pos_in = []
        self.checked = checked
        self.switch = switch
        self.switch_seq = switch_seq
        self.success = success

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            gripper_visualizations=gripper_visualizations,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )
        # number of learning parameters without PD params.
        if self.action_space_def == "Impedance_K":
            self.control_spec = 36
        if self.action_space_def == "Impedance_KC":
            self.control_spec = 72
        if self.action_space_def == "Impedance_KCM":
            self.control_spec = 108

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 100.0 is provided if the peg is inside the plate's hole
              - Note that we enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        Un-normalized summed components if using reward shaping:

            - ????

        Note that the final reward is normalized and scaled by reward_scale / 5.0 as
        well so that the max score is equal to reward_scale

        """
        #   TODO - reward(self, action=None) - change this function
        reward = 0
        time_factor = (self.horizon - self.timestep) / self.horizon
        # Right location and angle
        if self._check_success() and self.checked == (self.num_via_points-1):
            # reward = self.horizon * time_factor

            self.success += 1
            if self.success == 2:
                S=1
            return reward

        # use a shaping reward
        if self.reward_shaping:
            # Grab relevant values
            t, d, cos = self._compute_orientation()
            # Reach a terminal state as quickly as possible

            # reaching reward
            reward += self.r_reach*5*cos #* time_factor

            # Orientation reward
            reward += 1 - np.tanh(d)
            # reward += 1 - np.tanh(np.abs(t))
            reward += cos

        # if we're not reward shaping, we need to scale our sparse reward so that the max reward is identical
        # to its dense version
        else:
            reward *= 5.0

        if self.reward_scale is not None:
            reward *= self.reward_scale

        if (self.checked == (self.num_via_points-1)
            and ((abs(self.hole_pos[0] - self.peg_pos[0]) > 0.014
            or abs(self.hole_pos[1] - self.peg_pos[1]) > 0.014)
            and self.peg_pos[2] < self.mujoco_arena.table_offset[2] + 0.1)
            or self.horizon - self.timestep == 1
        ):
            reward = 0*-self.horizon/3
            self.checked = (self.num_via_points-2)
            self.switch = 0
            self.switch_seq = 0
            self.success = 0
            # self.trans *= 3
            self.reset_via_point()
            self.built_min_jerk_traj()


        return reward

    def on_peg(self):

        res = False
        if (
                    abs(self.hole_pos[0] - self.peg_pos[0]) < 0.002
                    and abs(self.hole_pos[1] - self.peg_pos[1]) < 0.002
                    and abs(self.hole_pos[1] - self.peg_pos[1]) + abs(self.hole_pos[0] - self.peg_pos[0]) < 0.04
                    and self.peg_pos[2] < self.mujoco_arena.table_offset[2] + 0.03
        ):
            res = True
        return res

    def clear_objects(self, obj):
        """
        Clears objects without the name @obj out of the task space. This is useful
        for supporting task modes with single types of objects, as in
        @self.single_object_mode without changing the model definition.

        Args:
            obj (str): Name of object to keep in the task space
        """
        for obj_name, obj_mjcf in self.mujoco_objects.items():
            if obj_name == obj:
                continue
            else:
                sim_state = self.sim.get_state()
                # print(self.sim.model.get_joint_qpos_addr(obj_name))
                sim_state.qpos[self.sim.model.get_joint_qpos_addr(obj_name + "_jnt0")[0]] = 10
                self.sim.set_state(sim_state)
                self.sim.forward()

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        # super()._load_model()
        # self.model = MujocoWorldBase()
        # # Verify the correct robot has been loaded
        # assert isinstance(self.robots[0], SingleArm), \
        #     "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))
        #
        # # Adjust base pose accordingly
        # xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        # xpos = (-0.1, 0, 0)
        # self.robots[0].robot_model.set_base_xpos(xpos)
        #
        # # load model for table top workspace
        # self.mujoco_arena = TableArena(
        #     table_full_size=self.table_full_size, table_friction=self.table_friction
        # )
        # if self.use_indicator_object:
        #     self.mujoco_arena.add_pos_indicator()
        #
        # # The sawyer robot has a pedestal, we want to align it with the table
        # self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])
        #
        # self.model.merge(self.mujoco_arena)
        # for robot in self.robots:
        #     self.model.merge(robot.robot_model)
        #
        # self.peg = CylinderObject(
        #     name="peg",
        #     size_min=(self.peg_radius[0], self.peg_length),
        #     size_max=(self.peg_radius[1], self.peg_length),
        #     # material=greenwood,
        #     rgba=[0, 1, 0, 1],
        # )
        #
        #
        # # Load peg object
        # self.peg_obj = self.peg.get_collision(site=True)
        # self.peg_obj.set("pos", array_to_string((0, 0, self.peg_length)))
        # self.model.merge_asset(self.peg)
        #
        # # Depending on env configuration, append appropriate objects to arms
        # self.model.worldbody.find(".//body[@name='{}']"
        #                           .format(self.robots[0].robot_model.eef_name)).append(self.peg_obj)
        #
        # self.hole = PlateWith5mmHoleObject(
        #     name="hole",
        # )
        # # object placement initializer
        # # treat sampling of each type of nut differently since we require different
        # # sampling ranges for each
        # self.placement_initializer = SequentialCompositeSampler()
        # self.placement_initializer.sample_on_top(
        #     "hole",
        #     surface_name="table",
        #     x_range=[-0.3, -0.2],#[-0.115, -0.11],
        #     y_range=[-0.11, 0.225],#[0.11, 0.225],
        #     rotation=None,
        #     rotation_axis='z',
        #     z_offset=0.02,
        #     ensure_object_boundary_in_range=False,
        # )
        #
        #
        # # Load hole object
        # self.hole_obj = self.hole.get_collision(site=True)
        # # self.hole_obj.set("name", "hole")
        # self.hole_obj.set("quat", "0 0 0.707 0.707")
        # self.hole_obj.set("pos", "0.11 0 0.17")
        # self.model.merge_asset(self.hole)
        #
        # self.model.worldbody.find(".//body[@name='table']").append(self.hole_obj)
        # # set positions of objects
        # # self.model.place_objects()
        #
        # # super()._load_model()

        #######################################################################################################
        super()._load_model()


        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], SingleArm), \
            "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        xpos = (-0.1, 0, 0)
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        self.peg = CylinderObject(
            name="peg",
            size_min=(self.peg_radius[0], self.peg_length),
            size_max=(self.peg_radius[1], self.peg_length),
            # material=greenwood,
            rgba=[0, 1, 0, 1],
        )


        # Load peg object
        self.peg_obj = self.peg.get_collision(site=True)
        self.peg_obj.set("pos", array_to_string((0, 0, self.peg_length)))


        # self.peg = CylinderObject(
        #     name="peg",
        #     size_min=(self.box_face, self.box_depth*50),
        #     size_max=(self.box_face, self.box_depth*50),
        #     rgba=[1, 0, 0, 1],
        #     # friction=fric,
        # )

        # self.obj_name = "cylinder"
        self.hole = PlateWith5mmHoleObject(
            # name='hole',
        )

        # self.dist_error=self.dist_error/2

        # self.objective = "middle_"+str(self.obj_name)

        # define mujoco objects
        # lst = []
        # lst.append(("peg", self.peg))
        # lst.append(("hole", self.hole))
        #
        # self.mujoco_objects = OrderedDict(lst)
        # self.n_objects = len(self.mujoco_objects)

        self.mujoco_gripper_objects = OrderedDict([(self.peg_obj, self.peg)])
        self.mujoco_objects = OrderedDict([("hole", self.hole)])

        self.n_objects = len(self.mujoco_objects)

        self.model = ManipulationTaskNoGripperFunc(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.mujoco_objects,
            mujoco_gripper_objects=self.mujoco_gripper_objects,
            visual_objects=None,
            initializer=self.placement_initializer,
        )

        # set positions of objects
        self.model.place_objects()

        super()._load_model()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        #   TODO - _get_reference(self)- check this function
        super()._get_reference()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id("hole")
        self.peg_body_id = self.sim.model.body_name2id("peg")

        # id of grippers for contact checking
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.important_geoms["left_finger"]
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.robots[0].gripper.important_geoms["right_finger"]
        ]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        #   TODO - _reset_internal(self) - change this function?!!! maybe
        super()._reset_internal()
        self.checked = 0
        self.switch = 0
        self.switch_seq = 0
        self.success = 0
        # self.trans = 0.15
        self.reset_via_point()

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:

            `'robot-state'`: contains robot-centric information.

            `'object-state'`: requires @self.use_object_obs to be True. Contains object-centric information.

            `'image'`: requires @self.use_camera_obs to be True. Contains a rendered frame from the simulation.

            `'depth'`: requires @self.use_camera_obs and @self.camera_depth to be True.
            Contains a rendered depth map from the simulation

        Returns:
            OrderedDict: Observations from the environment
        """
        #   TODO - _get_observation - change this function
        di = super()._get_observation()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix
            pr = self.robots[0].robot_model.naming_prefix

            # remember the keys to collect into object info
            object_state_keys = []

            # for conversion to relative gripper frame
            gripper_pose = T.pose2mat((di[pr + "eef_pos"], di[pr + "eef_quat"]))
            world_pose_in_gripper = T.pose_inv(gripper_pose)

            # position and rotation of peg and hole
            hole_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
            hole_quat = T.convert_quat(
                self.sim.data.body_xquat[self.hole_body_id], to="xyzw"
            )
            di["hole_pos"] = hole_pos
            di["hole_quat"] = hole_quat

            peg_pos = self.peg_pos  #np.array(self.sim.data.body_xpos[self.peg_body_id])
            peg_quat = T.convert_quat(
                self.sim.data.body_xquat[self.peg_body_id], to="xyzw"
            )
            di["peg_to_hole"] = peg_pos - hole_pos
            di["peg_quat"] = peg_quat

            # Relative orientation parameters
            t, d, cos = self._compute_orientation()
            di["angle"] = cos
            # di["t"] = t
            # del di
            # di = {}
            di["d"] = 1

            di["object-state"] = np.concatenate(
                [
                    di["hole_pos"],
                    # di["hole_quat"],
                    # di["peg_to_hole"],
                    # di["peg_quat"],
                    # [di["angle"]],
                    # [di["t"]],
                    # [di["d"]],
                ]
            )

        return di

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        #   TODO - _check_success(self) - change this function
        #   calculat pegs end position.
        self.r_reach = 0
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos_center = self.sim.data.body_xpos[self.peg_body_id]
        handquat = T.convert_quat(self.sim.data.get_body_xquat("robot0_right_hand"), to="xyzw")
        handDCM = T.quat2mat(handquat)
        self.peg_pos = peg_pos_center + (handDCM @ [0, 0, 2*self.peg_length]).T

        self.hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)

        dist = np.linalg.norm(self.peg_pos - self.hole_pos)
        self.r_reach = 1 - np.tanh(1.0 * dist)
        self.objects_on_pegs = int(self.on_peg() and self.r_reach > 0.96)

        return np.sum(self.objects_on_pegs) > 0

        # t, d, cos = self._compute_orientation()
        # if (d < 0.099 and -0.12 <= t <= 0.12 and cos > 0.95):
        #     D=1
        # return d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95

        # return d < 0.099 and -0.12 <= t <= 0.08 and cos > 0.95

    def _visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to cube
        if self.robots[0].gripper_visualization:
            # find closest object
            square_dist = lambda x: np.sum(
                np.square(x - self.sim.data.get_site_xpos(self.robots[0].gripper.visualization_sites["grip_site"]))
            )
            dists = np.array(list(map(square_dist, self.sim.data.site_xpos)))
            dists[self.robots[0].eef_site_id] = np.inf  # make sure we don't pick the same site
            dists[self.robots[0].eef_cylinder_id] = np.inf
            ob_dists = dists[self.object_site_ids]  # filter out object sites we care about
            min_dist = np.min(ob_dists)
            ob_id = np.argmin(ob_dists)

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(min_dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.robots[0].eef_site_id] = rgba

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"

    def _compute_orientation(self):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.

        Returns:
            3-tuple:

                - (float): parallel distance
                - (float): perpendicular distance
                - (float): angle
        """
        #   calculat pegs end position.
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos_center = self.sim.data.body_xpos[self.peg_body_id]
        handquat = T.convert_quat(self.sim.data.get_body_xquat("robot0_right_hand"), to="xyzw")
        handDCM = T.quat2mat(handquat)
        peg_pos = peg_pos_center + (handDCM @ [0, 0, self.peg_length]).T

        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)

        v = peg_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)
        center = hole_pos + hole_mat @ np.array([0.1, 0, 0])

        t = (center - peg_pos) @ v / (np.linalg.norm(v) ** 2)
        d = np.linalg.norm(np.cross(v, peg_pos - center)) / np.linalg.norm(v)

        hole_normal = hole_mat @ np.array([0, 0, 1])
        return (
            t,
            d,
            abs(
                np.dot(hole_normal, v) / np.linalg.norm(hole_normal) / np.linalg.norm(v)
            ),
        )

    def reset_via_point(self):

        added0 = 0.07
        added1 = 0.0
        # added2 = 0.0
        pos_via_point_0 = deepcopy(self.sim.data.get_body_xpos("hole")) #deepcopy(self.sim.data.get_site_xpos('middle_cylinder'))
        pos_via_point_0[2] += added0
        pos_via_point_1 = deepcopy(self.sim.data.get_body_xpos("hole")) #deepcopy(self.sim.data.get_site_xpos(self.sim.model.site_names[10]))
        pos_via_point_1[2] -= added1
        # pos_via_point_2 = deepcopy(self.sim.data.get_body_xpos("hole")) #deepcopy(self.sim.data.get_site_xpos(self.sim.model.site_names[10]))
        # pos_via_point_2[2] -= added2

        angle_desired = T.quat2axisangle(self.sim.data.get_body_xquat("hole"))
        angle_desired = np.abs(angle_desired) % (np.pi / 2)
        angle_desired[0] = np.pi

        # trans_error=np.array([-1,1,0])* self.dist_error # fixed error
        correct_error = np.array([0.003, 0.00255, 0.0])*0 # [0.002, -0.00355, 0.0] -> for video [0.003, -0.00355, 0.0]
        # print(trans_error)
        trans_error = ((np.random.rand(3) - 0.5) * 2) * self.dist_error # chenged error
        trans_error[2]=0
        angle_error=((np.random.rand(3) - 0.5)*2)*(np.pi/2)*self.angle_error

        via_point_0 = np.concatenate((pos_via_point_0+trans_error+correct_error, angle_desired+angle_error), axis=-1)
        via_point_1 = np.concatenate((pos_via_point_1+trans_error+correct_error, angle_desired+angle_error), axis=-1)
        # via_point_2 = np.concatenate((pos_via_point_2+trans_error+correct_error, angle_desired+angle_error), axis=-1)

        self.via_points['p0'] = via_point_0
        self.via_points['p1'] = via_point_1
        # self.via_points['p2'] = via_point_2
        self.num_via_points = 2
        self.first_via_points = deepcopy(self.via_points)
        self.pos_in = deepcopy([self.sim.data.body_xpos[self.sim.model.body_name2id("peg")],
                       T.mat2euler(self.sim.data.get_body_xmat("robot0_right_hand"))])

    def _switch_via_point(self):
        dist = np.linalg.norm(self.peg_pos - self.hole_pos)
        self.r_reach = 1 - np.tanh(2.0 * dist)
        return int(    abs(self.hole_pos[0] - self.peg_pos[0]) < 0.0002
                    and abs(self.hole_pos[1] - self.peg_pos[1]) < 0.0002
                    and self.r_reach > 0.86
        )

    # def _peg_pose_in_hole_frame(self):
    #     """
    #     A helper function that takes in a named data field and returns the pose of that
    #     object in the base frame.
    #
    #     Returns:
    #         np.array: (4,4) matrix corresponding to the pose of the peg in the hole frame
    #     """
    #     # World frame
    #     peg_pos_in_world = self.sim.data.get_body_xpos("peg")
    #     peg_rot_in_world = self.sim.data.get_body_xmat("peg").reshape((3, 3))
    #     peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)
    #
    #     # World frame
    #     hole_pos_in_world = self.sim.data.get_body_xpos("hole")
    #     hole_rot_in_world = self.sim.data.get_body_xmat("hole").reshape((3, 3))
    #     hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)
    #
    #     world_pose_in_hole = T.pose_inv(hole_pose_in_world)
    #
    #     peg_pose_in_hole = T.pose_in_A_to_pose_in_B(
    #         peg_pose_in_world, world_pose_in_hole
    #     )
    #     return peg_pose_in_hole

    # def staged_rewards(self):
    #     """
    #     Calculates staged rewards based on current physical states.
    #     Stages consist of reaching, grasping, lifting, and hovering.
    #
    #     Returns:
    #         4-tuple:
    #
    #             - (float) reaching reward
    #             - (float) grasping reward
    #             - (float) lifting reward
    #             - (float) hovering reward
    #     """
    #     #   TODO - staged_rewards(self) - change this function
    #     reach_mult = 0.1
    #     grasp_mult = 0.35
    #     lift_mult = 0.5
    #     hover_mult = 0.7
    #
    #     # filter out objects that are already on the correct pegs
    #     names_to_reach = []
    #     objs_to_reach = []
    #     geoms_to_grasp = []
    #     geoms_by_array = []
    #
    #     for i in range(len(self.ob_inits)):
    #         if self.objects_on_pegs[i]:
    #             continue
    #         obj_str = str(self.item_names[i]) + "0"
    #         names_to_reach.append(obj_str)
    #         objs_to_reach.append(self.obj_body_id[obj_str])
    #         geoms_to_grasp.extend(self.obj_geom_id[obj_str])
    #         geoms_by_array.append(self.obj_geom_id[obj_str])
    #
    #     ### reaching reward governed by distance to closest object ###
    #     r_reach = 0.
    #     if len(objs_to_reach):
    #         # reaching reward via minimum distance to the handles of the objects (the last geom of each nut)
    #         geom_ids = [elem[-1] for elem in geoms_by_array]
    #         target_geom_pos = self.sim.data.geom_xpos[geom_ids]
    #         gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
    #         dists = np.linalg.norm(
    #             target_geom_pos - gripper_site_pos.reshape(1, -1), axis=1
    #         )
    #         r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult
    #
    #     ### grasping reward for touching any objects of interest ###
    #     touch_left_finger = False
    #     touch_right_finger = False
    #     for i in range(self.sim.data.ncon):
    #         c = self.sim.data.contact[i]
    #         if c.geom1 in geoms_to_grasp:
    #             if c.geom2 in self.l_finger_geom_ids:
    #                 touch_left_finger = True
    #             if c.geom2 in self.r_finger_geom_ids:
    #                 touch_right_finger = True
    #         elif c.geom2 in geoms_to_grasp:
    #             if c.geom1 in self.l_finger_geom_ids:
    #                 touch_left_finger = True
    #             if c.geom1 in self.r_finger_geom_ids:
    #                 touch_right_finger = True
    #     has_grasp = touch_left_finger and touch_right_finger
    #     r_grasp = int(has_grasp) * grasp_mult
    #
    #     ### lifting reward for picking up an object ###
    #     r_lift = 0.
    #     table_pos = np.array(self.sim.data.body_xpos[self.table_body_id])
    #     if len(objs_to_reach) and r_grasp > 0.:
    #         z_target = table_pos[2] + 0.2
    #         object_z_locs = self.sim.data.body_xpos[objs_to_reach][:, 2]
    #         z_dists = np.maximum(z_target - object_z_locs, 0.)
    #         r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
    #                 lift_mult - grasp_mult
    #         )
    #
    #     ### hover reward for getting object above peg ###
    #     r_hover = 0.
    #     if len(objs_to_reach):
    #         r_hovers = np.zeros(len(objs_to_reach))
    #         for i in range(len(objs_to_reach)):
    #             if names_to_reach[i].startswith(self.item_names[0]):
    #                 peg_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])[:2]
    #             elif names_to_reach[i].startswith(self.item_names[1]):
    #                 peg_pos = np.array(self.sim.data.body_xpos[self.peg2_body_id])[:2]
    #             else:
    #                 raise Exception(
    #                     "Got invalid object to reach: {}".format(names_to_reach[i])
    #                 )
    #             ob_xy = self.sim.data.body_xpos[objs_to_reach[i]][:2]
    #             dist = np.linalg.norm(peg_pos - ob_xy)
    #             r_hovers[i] = r_lift + (1 - np.tanh(10.0 * dist)) * (
    #                     hover_mult - lift_mult
    #             )
    #         r_hover = np.max(r_hovers)
    #
    #     return r_reach, r_grasp, r_lift, r_hover

    # def _dict_action(self, action):
    #     """
    #     ????????????
    #     action_space_def - control how many parameters to learn.
    #     'Impedance_K' = matrix K (36) + PD (4) = 40
    #     'Impedance_KC' = matrix K and C (36+36) + PD (4) = 76
    #     'Impedance_KCM' = matrix K, C, M (36+36+36) + PD (4) = 112
    #     """
    #     # number_of_points = self.num_via_points
    #     # number_of_param = self.control_spec
    #
    #     if self.action_space_def == "Impedance_K":
    #         self.control_spec = 36
    #         self.action_dict['Imp' + str(1)] = action
    #     if self.action_space_def == "Impedance_KC":
    #         self.control_spec = 72
    #         self.action_dict['Imp' + str(1)] = action
    #     if self.action_space_def == "Impedance_KCM":
    #         self.control_spec = 108
    #         self.action_dict['Imp' + str(1)] = action
    #     if self.action_space_def == "Impedance_KC_PD":
    #         self.control_spec = 76
    #         self.action_dict['PD' + str(0)] = action[0:2]
    #         self.action_dict['PD' + str(1)] = action[2:4]
    #         self.action_dict['Imp' + str(1)] = action[4:]
    #         # for i in range(self.num_via_points):
    #         #   self.action_dict['PD'+str(i)]=action[i*relevent_param:i*relevent_param+2]
    #         #  self.action_dict['Imp' + str(i)] = action[i * relevent_param+2:(i+1)* relevent_param]