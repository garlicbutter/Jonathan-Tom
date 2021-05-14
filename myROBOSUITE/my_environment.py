from collections import OrderedDict
import numpy as np
import my_object
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor

class MyEnv(SingleArmEnv):
    '''
    my environment
    '''
    def __init__(
        self,
        robots,
        env_configuration="default",
        task_configs=None,
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # task configuration
        self.task_configs = task_configs

        # list of MujocoObject that will be usedf in the task
        self.objects_of_interest = []

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
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
        pass
    
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize plate with the corresponding option
        if self.task_configs['board'] == 'Hole12mm':
            self.plate = my_object.GMC_Assembly_Object(name="plate",)
            if not any(isinstance(x, my_object.GMC_Assembly_Object) for x in self.objects_of_interest):
                self.objects_of_interest.append(self.plate)
        
        elif self.task_configs['board'] == 'Hole9mm':
            pass
        elif self.task_configs['board'] == 'Hole6mm':
            pass

        if self.task_configs['peg'] == '16mm':
            self.peg = my_object.Round_peg_16mm_Object(name="peg",)
            if not any(isinstance(x, my_object.Round_peg_16mm_Object) for x in self.objects_of_interest):
                self.objects_of_interest.append(self.peg)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            # self.placement_initializer.add_objects(self.plate)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.objects_of_interest,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.objects_of_interest,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        if self.plate is not None:
            self.plate_body_id = self.sim.model.body_name2id(self.plate.root_body)
        
        if self.peg is not None:
            self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            
            sensors = []
            names = []
            # plate-related observables
            if self.plate is not None:
                @sensor(modality=modality)
                def plate_pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.plate_body_id])

                @sensor(modality=modality)
                def plate_quat(obs_cache):
                    return convert_quat(np.array(self.sim.data.body_xquat[self.plate_body_id]), to="xyzw")

                @sensor(modality=modality)
                def gripper_to_plate_pos(obs_cache):
                    return obs_cache[f"{pf}eef_pos"] - obs_cache["plate_pos"] if \
                        f"{pf}eef_pos" in obs_cache and "plate_pos" in obs_cache else np.zeros(3)

                sensors_plate = [plate_pos, plate_quat, gripper_to_plate_pos]
                names_plate = [s.__name__ for s in sensors_plate]
                sensors.extend(sensors_plate)
                names.extend(names_plate)
            
            # peg-related observables
            if self.peg is not None:
                @sensor(modality=modality)
                def peg_pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[self.peg_body_id])

                @sensor(modality=modality)
                def peg_quat(obs_cache):
                    return convert_quat(np.array(self.sim.data.body_xquat[self.peg_body_id]), to="xyzw")

                @sensor(modality=modality)
                def gripper_to_peg_pos(obs_cache):
                    return obs_cache[f"{pf}eef_pos"] - obs_cache["peg_pos"] if \
                        f"{pf}eef_pos" in obs_cache and "peg_pos" in obs_cache else np.zeros(3)

                sensors_peg = [peg_pos, peg_quat, gripper_to_peg_pos]
                names_peg = [s.__name__ for s in sensors_peg]
                sensors.extend(sensors_peg)
                names.extend(names_peg)

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        pass

    def _check_success(self):
        pass