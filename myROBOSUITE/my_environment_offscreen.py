from collections import OrderedDict
import numpy as np
import my_object
from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.models.objects import PlateWithHoleObject



class MyEnvOffScreen(SingleArmEnv):
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
        use_camera_obs=False,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=20,
        ignore_done=True,
        hard_reset=True,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # task configuration
        self.task_configs = task_configs

        # list of MujocoObject that will be usedf in the task
        self.objects_of_interest = [] # for object
        self.objectsName_of_interest = [] # for name
        self.objectsXrange_of_interest = [] # x range for placement
        self.objectsYrange_of_interest = [] # y range for placement

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
            control_freq=control_freq,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
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


        # Load hole object
        # register object with the corresponding option (objectClass, name, xrange, yrange)
        if self.task_configs['board'] == 'GMC_assembly':
            self.register_object(my_object.GMC_Assembly_Object,'plate',xrange=[0,0],yrange=[0,0])
        if self.task_configs['board'] == 'GMC_plate':
            self.register_object(my_object.GMC_Plate_Object,'plate',xrange=[0,0],yrange=[0,0])
        if self.task_configs['board'] == 'Square_hole_16mm':
            self.register_object(my_object.square_hole_16mm,'plate',xrange=[0,0],yrange=[0,0])

        # Load peg object
        if self.task_configs['peg'] == '16mm':
            self.register_object(my_object.Round_peg_16mm_Object,'peg',xrange=[-0.1,-0.13],yrange=[0.3,0.33])
        elif self.task_configs['peg'] == '12mm':
            self.register_object(my_object.Round_peg_12mm_Object,'peg',xrange=[-0.1,-0.13],yrange=[0.3,0.33])
        elif self.task_configs['peg'] == '9mm':
            raise NotImplementedError
        elif self.task_configs['peg'] == 'cylinder_16mm':
            from robosuite.models.objects.primitive import CylinderObject
            self.peg = CylinderObject('peg',size=(0.007,0.025))
            self.objects_of_interest.append(self.peg)
            self.objectsName_of_interest.append('peg')
            self.objectsXrange_of_interest.append([-0.1,-0.13])
            self.objectsYrange_of_interest.append([0.3,0.33])

        # Create Sequential Sampler. The order is same as the order of register.
        # Create individual samplers per object
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for obj_name, default_xy_range in zip(self.objectsName_of_interest,zip(self.objectsXrange_of_interest ,self.objectsYrange_of_interest)):
            self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"{obj_name}Sampler",
                x_range=default_xy_range[0],
                y_range=default_xy_range[1],
                rotation=None,
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            )
        # Add objects to the sampler
        for obj_to_put, obj_name in zip(self.objects_of_interest,self.objectsName_of_interest):
            self.placement_initializer.add_objects_to_sampler(sampler_name=f"{obj_name}Sampler", mujoco_objects=obj_to_put)

        if self.task_configs['board'] == 'hole':
            self.plate = PlateWithHoleObject(name='plate')
            plate_obj = self.plate.get_obj()
            plate_obj.set("quat", "0 0 0 1")
            plate_obj.set("pos", "0 0 {}".format(self.table_offset[2]))
            self.objects_of_interest.append(self.plate)

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

    def register_object(self,objectClass,name,xrange=[0,0],yrange=[0,0]):
        # initialize object with the corresponding option
        exec('self.{} = objectClass(name=name)'.format(name))
        exec('self.objects_of_interest.append(self.{})'.format(name))
        self.objectsName_of_interest.append(name)
        self.objectsXrange_of_interest.append(xrange)
        self.objectsYrange_of_interest.append(yrange)

    def visualize(self, vis_settings):
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        pass

    def get_ee_force(self):
        return self.ee_force