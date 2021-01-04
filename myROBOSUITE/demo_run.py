import numpy as np
import robosuite as suite
from robosuite import load_controller_config

# env setting
env_name = "Lift"
robot_type = "UR5e"
# Load the desired controller's default config as a dict
controller_name = 'JOINT_POSITION'
controller_configs = load_controller_config(default_controller=controller_name)
gripper_types = "default"
initialization_noise = "default"
table_full_size = (0.8, 0.8, 0.05)
table_friction = (1., 5e-3, 1e-4)
use_camera_obs = False
use_object_obs = False
reward_scale = 1.0
reward_shaping = False
placement_initializer = None
has_renderer = True
has_offscreen_renderer = False
render_camera = "frontview"
render_collision_mesh = False
render_visual_mesh = True
render_gpu_device_id = -1
control_freq = 20
horizon = 1000
ignore_done = False
hard_reset = True
camera_names = "agentview"
camera_heights = 256
camera_widths = 256
camera_depths = False


# create environment instance
env = suite.make(
    env_name=env_name,
    robots=robot_type,
    controller_configs=controller_configs,
    gripper_types=gripper_types,
    initialization_noise=initialization_noise,
    table_full_size=table_full_size,
    table_friction=table_friction,
    use_camera_obs=use_camera_obs,
    use_object_obs=use_object_obs,
    reward_scale=reward_scale,
    reward_shaping=reward_shaping,
    placement_initializer=placement_initializer,
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

# reset the environment
env.reset()


# action setting

action_dim = 6
gripper_dim = 1
neutral = np.zeros(action_dim + gripper_dim)


# simulate
for i in range(1000):
    action = neutral.copy()
    action = [0, 0, 0, 0, 0, 0, 0]
    env.step(action)
    env.render()


# Shut down this env before starting the next test
env.close()
