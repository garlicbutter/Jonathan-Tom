"""
This script shows how to adapt an environment to be compatible
with the OpenAI Gym-style API. This is useful when using
learning pipelines that require supporting these APIs.

For instance, this can be used with OpenAI Baselines
(https://github.com/openai/baselines) to train agents
with RL.


We base this script off of some code snippets found
in the "Getting Started with Gym" section of the OpenAI 
gym documentation.

The following snippet was used to demo basic functionality.

    import gym
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

To adapt our APIs to be compatible with OpenAI Gym's style, this script
demonstrates how this can be easily achieved by using the GymWrapper.
"""
from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize
import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper
import numpy as np
from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import logger
from robosuite import load_controller_config


# def evaluate(model: "base_class.BaseAlgorithm",
#              env: Union[gym.Env, VecEnv],
#             n_eval_episodes: int = 10,
#             deterministic: bool = True,
#             render: bool = False,
#             callback: Optional[Callable] = None,
#             reward_threshold: Optional[float] = None,
#             return_episode_rewards: bool = False,
#         ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
#     """
#     Runs policy for ``n_eval_episodes`` episodes and returns average reward.
#     This is made to work only with one env.
#
#     :param model: The RL agent you want to evaluate.
#     :param env: The gym environment. In the case of a ``VecEnv``
#         this must contain only one environment.
#     :param n_eval_episodes: Number of episode to evaluate the agent
#     :param deterministic: Whether to use deterministic or stochastic actions
#     :param render: Whether to render the environment or not
#     :param callback: callback function to do additional checks,
#         called after each step.
#     :param reward_threshold: Minimum expected reward per episode,
#         this will raise an error if the performance is not met
#     :param return_episode_rewards: If True, a list of reward per episode
#         will be returned instead of the mean.
#     :return: Mean reward per episode, std of reward per episode
#         returns ([float], [int]) when ``return_episode_rewards`` is True
#     """
#     if isinstance(env, VecEnv):
#         assert env.num_envs == 1, "You must pass only one environment when using this function"
#
#     episode_rewards, episode_lengths = [], []
#     for i in range(n_eval_episodes):
#         # Avoid double reset, as VecEnv are reset automatically
#         if not isinstance(env, VecEnv) or i == 0:
#             obs = env.reset()
#         done, state = False, None
#         episode_reward = 0.0
#         episode_length = 0
#         while not done:
#             action, state = model.predict(obs, state=state, deterministic=deterministic)
#             obs, reward, done, _info = env.step(action)
#             episode_reward += reward
#             if callback is not None:
#                 callback(locals(), globals())
#             episode_length += 1
#             if render:
#                 env.render()
#         episode_rewards.append(episode_reward)
#         episode_lengths.append(episode_length)
#         print(f"episode number: {i},reward: {episode_reward}, episode lenght: {_info.get('time')} ")
#     mean_reward = np.mean(episode_rewards)
#     std_reward = np.std(episode_rewards)
#     if reward_threshold is not None:
#         assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
#     if return_episode_rewards:
#         return episode_rewards, episode_lengths
#     return mean_reward, std_reward

def train():
    """
    Train PPO2 model for Mujoco environment, for testing purposes
    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    """
    def make_env():
        # Load the desired controller's default config as a dict
        # config = load_controller_config(default_controller='OSC_POSE')
        control_param = dict(type='IMPEDANCE_PB', input_max=1, input_min=-1,
                             output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                             output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], kp=150, damping_ratio=1,
                             impedance_mode='fixed', kp_limits=[0, 300], damping_ratio_limits=[0, 10],
                             position_limits=None,
                             orientation_limits=None, uncouple_pos_ori=True, control_delta=True, interpolation=None,
                             ramp_ratio=0.2)
        # robot_param = dict(robot_type='UR5e', idn=0, initial_qpos=np.array([-0.47, -1.735,  2.48, -2.275, -1.59, -0.42])
        #                    , initialization_noise=None,)

        # Notice how the environment is wrapped by the wrapper
        env = GymWrapper(
            suite.make(
                "PegInHole",
                robots="UR5e",  # use UR5e robot
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=True,  ##True  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                ignore_done=False,
                horizon=500,
                control_freq=20,  # control should happen fast enough so that simulation looks smooth
                controller_configs=control_param
            )
        )
        env_out = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), info_keywords=('is_success','time',), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([lambda:make_env])
    env = VecNormalize.load(env)

    policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU, net_arch=[32, 32])
    model = PPO(MlpPolicy, env, verbose=2,policy_kwargs=policy_kwargs, n_epochs=2, batch_size=3)

    model.learn(total_timesteps=1000)

    return model, env


def main():
    """
    Runs the test
    """
    logger.configure()
    model, env = train()

    if args.play:
        logger.log("Running trained model")
    model.save(str("testing"))

if __name__ == "__main__":
    main()

    # env = VecEnv([lambda: env])
    policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU, net_arch=[32, 32])
    model = PPO(MlpPolicy, env, verbose=2,policy_kwargs=policy_kwargs, n_epochs=2, batch_size=3)

    # Random Agent, before training
    # mean_reward, std_reward = evaluate(model, env, n_eval_episodes=5, render=True)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    model.learn(total_timesteps=1000)#, eval_env=evaluate(model, env, n_eval_episodes=10))
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    mean_reward, std_reward = evaluate(model, env, n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    model.save(str("testing"))
