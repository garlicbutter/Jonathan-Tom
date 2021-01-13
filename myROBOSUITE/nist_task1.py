import robosuite as suite
from robosuite.wrappers import GymWrapper
import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.ppo import MlpPolicy
# from stable_baselines3.common.evaluation import evaluate_policy
from robosuite import load_controller_config


if __name__ == "__main__":

    # Load the desired controller's default config as a dict
    # config = load_controller_config(default_controller='OSC_POSE')
    config = {'type': 'IMPEDANCE_PB', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
              'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping_ratio': 1,
              'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10],
              'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True,
              'interpolation': None, 'ramp_ratio': 0.2}

    # Notice how the environment is wrapped by the wrapper
    env = suite.make(
        "PegInHole",
        robots="UR5e",  # use Sawyer robot
        use_camera_obs=False,  # do not use pixel observations
        has_offscreen_renderer=False,  # not needed since not using pixel obs
        has_renderer=True,  # make sure we can render to the screen
        reward_shaping=True,  # use dense rewards
        control_freq=20,  # control should happen fast enough so that simulation looks smooth
        controller_configs=config
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
