from collections import OrderedDict
import random
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.robot_env import RobotEnv
from robosuite.models import MujocoWorldBase
from robosuite.robots import SingleArm
from robosuite.controllers import controller_factory, load_controller_config
from robosuite.models.arenas import TableArena, EmptyArena
from robosuite.models.objects import PlateWithHoleObject, ElectricPanel, CylinderObject
from robosuite.models.tasks import ManipulationTask, SequentialCompositeSampler
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
        peg_radius=(0.015, 0.03),
        peg_length=0.13,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
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
    ):
        # First, verify that only one robot is being inputted
        self._check_robot_configuration(robots)

        # # task settings
        # self.single_object_mode = single_object_mode
        # self.nut_to_id = {"square": 0, "round": 1}
        # if nut_type is not None:
        #     assert (
        #             nut_type in self.nut_to_id.keys()
        #     ), "invalid @nut_type argument - choose one of {}".format(
        #         list(self.nut_to_id.keys())
        #     )
        #     self.nut_id = self.nut_to_id[nut_type]  # use for convenient indexing
        # self.obj_to_use = None

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # set via points:
        self.via_point = {"pos": [0, 0, 0], "gripper": False}
        self.reach = 0
        self.grasp = 0
        self.lift = 0

        # Save peg specs
        self.peg_radius = peg_radius
        self.peg_length = peg_length

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

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 5.0 is provided if the peg is inside the plate's hole
              - Note that we enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arms to approach each other
            - Perpendicular Distance: in [0,1], to encourage the arms to approach each other
            - Parallel Distance: in [0,1], to encourage the arms to approach each other
            - Alignment: in [0, 1], to encourage having the right orientation between the peg and hole.
            - Placement: in {0, 1}, nonzero if the peg is in the hole with a relatively correct alignment

        Note that the final reward is normalized and scaled by reward_scale / 5.0 as
        well so that the max score is equal to reward_scale

        """
        reward = 0

        # Right location and angle
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            # Grab relevant values
            t, d, cos = self._compute_orientation()
            # reaching reward
            hole_pos = self.sim.data.body_xpos[self.hole_body_id]
            gripper_site_pos = self.sim.data.body_xpos[self.peg_body_id]
            dist = np.linalg.norm(gripper_site_pos - hole_pos)
            reaching_reward = 1 - np.tanh(1.0 * dist)
            reward += reaching_reward

            # Orientation reward
            reward += 1 - np.tanh(d)
            reward += 1 - np.tanh(np.abs(t))
            reward += cos

        # if we're not reward shaping, we need to scale our sparse reward so that the max reward is identical
        # to its dense version
        else:
            reward *= 5.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 5.0

        self.reset_via_point()

        return reward

    def staged_rewards(self):
        """
        Calculates staged rewards based on current physical states.
        Stages consist of reaching, grasping, lifting, and hovering.

        Returns:
            4-tuple:

                - (float) reaching reward
                - (float) grasping reward
                - (float) lifting reward
                - (float) hovering reward
        """

        reach_mult = 0.1
        grasp_mult = 0.35
        lift_mult = 0.5
        hover_mult = 0.7

        # filter out objects that are already on the correct pegs
        names_to_reach = []
        objs_to_reach = []
        geoms_to_grasp = []
        geoms_by_array = []

        for i in range(len(self.ob_inits)):
            if self.objects_on_pegs[i]:
                continue
            obj_str = str(self.item_names[i]) + "0"
            names_to_reach.append(obj_str)
            objs_to_reach.append(self.obj_body_id[obj_str])
            geoms_to_grasp.extend(self.obj_geom_id[obj_str])
            geoms_by_array.append(self.obj_geom_id[obj_str])

        ### reaching reward governed by distance to closest object ###
        r_reach = 0.
        if len(objs_to_reach):
            # reaching reward via minimum distance to the handles of the objects (the last geom of each nut)
            geom_ids = [elem[-1] for elem in geoms_by_array]
            target_geom_pos = self.sim.data.geom_xpos[geom_ids]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dists = np.linalg.norm(
                target_geom_pos - gripper_site_pos.reshape(1, -1), axis=1
            )
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        ### grasping reward for touching any objects of interest ###
        touch_left_finger = False
        touch_right_finger = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in geoms_to_grasp:
                if c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            elif c.geom2 in geoms_to_grasp:
                if c.geom1 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids:
                    touch_right_finger = True
        has_grasp = touch_left_finger and touch_right_finger
        r_grasp = int(has_grasp) * grasp_mult

        ### lifting reward for picking up an object ###
        r_lift = 0.
        table_pos = np.array(self.sim.data.body_xpos[self.table_body_id])
        if len(objs_to_reach) and r_grasp > 0.:
            z_target = table_pos[2] + 0.2
            object_z_locs = self.sim.data.body_xpos[objs_to_reach][:, 2]
            z_dists = np.maximum(z_target - object_z_locs, 0.)
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
                    lift_mult - grasp_mult
            )

        ### hover reward for getting object above peg ###
        r_hover = 0.
        if len(objs_to_reach):
            r_hovers = np.zeros(len(objs_to_reach))
            for i in range(len(objs_to_reach)):
                if names_to_reach[i].startswith(self.item_names[0]):
                    peg_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])[:2]
                elif names_to_reach[i].startswith(self.item_names[1]):
                    peg_pos = np.array(self.sim.data.body_xpos[self.peg2_body_id])[:2]
                else:
                    raise Exception(
                        "Got invalid object to reach: {}".format(names_to_reach[i])
                    )
                ob_xy = self.sim.data.body_xpos[objs_to_reach[i]][:2]
                dist = np.linalg.norm(peg_pos - ob_xy)
                r_hovers[i] = r_lift + (1 - np.tanh(10.0 * dist)) * (
                        hover_mult - lift_mult
                )
            r_hover = np.max(r_hovers)

        return r_reach, r_grasp, r_lift, r_hover

    def on_peg(self, obj_pos, peg_id):

        if peg_id == 0:
            peg_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])
        else:
            peg_pos = np.array(self.sim.data.body_xpos[self.peg2_body_id])
        res = False
        if (
                abs(obj_pos[0] - peg_pos[0]) < 0.03
                and abs(obj_pos[1] - peg_pos[1]) < 0.03
                and obj_pos[2] < self.mujoco_arena.table_offset[2] + 0.05
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
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], SingleArm), \
            "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Add arena and robot
        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=(0, 0, 0.82)
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # initialize objects of interest
        self.hole = PlateWithHoleObject(
            name="hole",
        )
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.peg = CylinderObject(
            name="peg",
            size_min=(self.peg_radius[0], self.peg_length),
            size_max=(self.peg_radius[1], self.peg_length),
            material=greenwood,
            rgba=[0, 1, 0, 1],
        )

        # Load hole object
        self.hole_obj = self.hole.get_collision(site=True)
        self.hole_obj.set("quat", "0 0 0.707 0.707")
        self.hole_obj.set("pos", "0.11 0 0.17")
        # self.model.merge_asset(self.hole)

        # Load peg object
        self.peg_obj = self.peg.get_collision(site=True)
        self.peg_obj.set("pos", array_to_string((0, 0, self.peg_length)))
        # self.model.merge_asset(self.peg)

        # self.mujoco_objects = OrderedDict([("peg", (self.peg))])
        # self.mujoco_objects = OrderedDict([("hole", self.hole)])
        # self.n_objects = len(self.mujoco_objects)

        # define mujoco objects
        lst = []
        lst.append(("peg", self.peg))
        lst.append(("hole", self.hole))


        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        # self.model = ManipulationTask(
        #     mujoco_arena=self.mujoco_arena,
        #     mujoco_robots=[robot.robot_model for robot in self.robots],
        #     mujoco_objects=self.peg_obj,
        #     visual_objects=None,
        # )

        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.mujoco_objects,
            visual_objects=None,
            # initializer=self.placement_initializer,
        )


        # set positions of objects
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
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
        super()._reset_internal()

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

            peg_pos = np.array(self.sim.data.body_xpos[self.peg_body_id])
            peg_quat = T.convert_quat(
                self.sim.data.body_xquat[self.peg_body_id], to="xyzw"
            )
            di["peg_to_hole"] = peg_pos - hole_pos
            di["peg_quat"] = peg_quat

            # Relative orientation parameters
            t, d, cos = self._compute_orientation()
            di["angle"] = cos
            di["t"] = t
            di["d"] = d

            di["object-state"] = np.concatenate(
                [
                    di["hole_pos"],
                    di["hole_quat"],
                    di["peg_to_hole"],
                    di["peg_quat"],
                    [di["angle"]],
                    [di["t"]],
                    [di["d"]],
                ]
            )

        return di

    def _check_success(self):
        """
        Check if all nuts have been successfully placed around their corresponding pegs.

        Returns:
            bool: True if all nuts are placed correctly
        """
        # remember objects that are on the correct pegs
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        gripper_site_ori = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id('gripper0_grip_site')].reshape([3, 3]))

        ###   checking success of reaching

        #   peg
        peg_pos = np.array(self.sim.data.body_xpos[self.peg_body_id])
        peg_ori = self.sim.data.get_body_xmat('peg')
        dist_xy = np.linalg.norm((gripper_site_pos - peg_pos)[:2])
        dist_z = np.linalg.norm((gripper_site_pos - peg_pos)[2])
        r_reach = 1 - np.tanh(10.0 * dist_xy)
        ori_reach = T.mat2euler(gripper_site_ori - peg_ori)[1]
        self.reach = int(dist_xy < 0.03 and ori_reach < 0.9 and dist_z < 0.09)

        # #   Nut
        # Nut_pos = self.sim.data.body_xpos[self.obj_body_id[str(self.item_names[1]) + "0"]]
        # Nut_ori = self.sim.data.get_body_xmat('RoundNut0')
        # dist_xy = np.linalg.norm((gripper_site_pos - Nut_pos)[:2])
        # dist_z = np.linalg.norm((gripper_site_pos - Nut_pos)[2])
        # r_reach = 1 - np.tanh(10.0 * dist_xy)
        # ori_reach = T.mat2euler(gripper_site_ori - Nut_ori)[1]
        # self.reach = int(dist_xy < 0.02 and ori_reach < 0.8 and dist_z < 0.09)
        #
        #
        ##   checking success of grasping:
        #   for peg:

        # geoms_to_grasp = self.obj_geom_id['peg']
        # touch_left_finger = False
        # touch_right_finger = False
        # for i in range(self.sim.data.ncon):
        #     c = self.sim.data.contact[i]
        #     if c.geom1 in geoms_to_grasp:
        #         if c.geom2 in self.l_finger_geom_ids:
        #             touch_left_finger = True
        #         if c.geom2 in self.r_finger_geom_ids:
        #             touch_right_finger = True
        #     elif c.geom2 in geoms_to_grasp:
        #         if c.geom1 in self.l_finger_geom_ids:
        #             touch_left_finger = True
        #         if c.geom1 in self.r_finger_geom_ids:
        #             touch_right_finger = True
        # has_grasp = touch_left_finger and touch_right_finger
        # self.grasp = int(has_grasp) * self.reach
        #
        # #   checking success of lifting + hovering above Nut:
        # Nut_pos = self.sim.data.body_xpos[self.obj_body_id[str(self.item_names[1]) + "0"]]
        # dist_xy = np.linalg.norm((gripper_site_pos - Nut_pos)[:2])
        # # dist_z = np.linalg.norm((gripper_site_pos - peg_pos)[2])
        # r_reach = 1 - np.tanh(10.0 * dist_xy)
        # ori_reach = T.mat2euler(gripper_site_ori - peg_ori)[1]
        # self.lift = int(dist_xy < 0.02 and ori_reach < 0.8)
        #
        # #   checking success of :
        #
        # #   checking success of inserting:
        # for i in range(len(self.ob_inits)):
        #     obj_str = str(self.item_names[i]) + "0"
        #     obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
        #     dist = np.linalg.norm(gripper_site_pos - obj_pos)
        #     r_reach = 1 - np.tanh(10.0 * dist)
        #     self.objects_on_pegs[i] = int(self.on_peg(obj_pos, i) and r_reach < 0.6)
        #
        # if self.single_object_mode > 0:
        #     return np.sum(self.objects_on_pegs) > 0  # need one object on peg
        #
        # # returns True if all objects are on correct pegs
        # self.reset_via_point()
        #
        # return np.sum(self.objects_on_pegs) == len(self.ob_inits)

        t, d, cos = self._compute_orientation()

        return d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95

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
            ob_dists = dists[
                self.object_site_ids
            ]  # filter out object sites we care about
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
        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_mat.shape = (3, 3)
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]

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

    def _peg_pose_in_hole_frame(self):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.

        Returns:
            np.array: (4,4) matrix corresponding to the pose of the peg in the hole frame
        """
        # World frame
        peg_pos_in_world = self.sim.data.get_body_xpos("peg")
        peg_rot_in_world = self.sim.data.get_body_xmat("peg").reshape((3, 3))
        peg_pose_in_world = T.make_pose(peg_pos_in_world, peg_rot_in_world)

        # World frame
        hole_pos_in_world = self.sim.data.get_body_xpos("hole")
        hole_rot_in_world = self.sim.data.get_body_xmat("hole").reshape((3, 3))
        hole_pose_in_world = T.make_pose(hole_pos_in_world, hole_rot_in_world)

        world_pose_in_hole = T.pose_inv(hole_pose_in_world)

        peg_pose_in_hole = T.pose_in_A_to_pose_in_B(
            peg_pose_in_world, world_pose_in_hole
        )
        return peg_pose_in_hole

    def reset_via_point(self):
        if not self.reach:
            self.via_point["pos"] = self.sim.data.get_body_xpos("peg")
            self.via_point["gripper"] = False
        elif not self.grasp:
            self.via_point["pos"] = self.sim.data.get_body_xpos("peg")
            self.via_point["gripper"] = True
        elif not self.lift:
            self.via_point["pos"] = self.sim.data.get_body_xpos("hole")
            self.via_point["pos"][2] += [1.0]
            self.via_point["gripper"] = True
