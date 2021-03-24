import gym

# from stable_baselines3.common.policies import
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3 import PPO
from robosuite.wrappers import GymWrapper
import robosuite as suite
import numpy as np
import time
import torch as tf
import argparse
from matplotlib import pyplot as plt
import csv
import os


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", help="n_steps")
    parser.add_argument("--net_arch", help="net_arch")
    parser.add_argument("--tb_log_name", help="tb_log_name")
    parser.add_argument("--action_space_def", help="action_space_def")
    parser.add_argument("--dist_error", help="dist_error")
    parser.add_argument("--angle_error", help="angle_error")
    parser.add_argument("--horizon", help="horizon")
    parser.add_argument("--control_freq", help="control_freq")
    parser.add_argument("--number_of_slices", help="number_of_slices")
    parser.add_argument("--num_iterations", help="num_iterations")
    parser.add_argument("--total_timesteps", help="total_timesteps")
    parser.add_argument("--model_name", help="model_name")
    parser.add_argument("--task_name", help="task_name")
    parser.add_argument("--tensorboard_log", help="tensorboard_log")
    parser.add_argument("--forces_round", help="forces_round")
    parser.add_argument("--PD_gain", help="PD_gain")

    args = parser.parse_args()

    horizon = int(args.horizon)
    control_freq = int(args.control_freq)
    number_of_slices = int(args.number_of_slices)
    num_iterations = int(args.num_iterations)

    # Create log dir
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    plotter = True
    # env = GymWrapper(suite.make(str(args.task_name), robots="UR5e", has_offscreen_renderer=False,
    #                             gripper_types="Robotiq85Gripper", has_renderer=True,
    #                             reward_shaping=True, control_freq=control_freq,
    #                             horizon=horizon))
    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "Stack",
            robots="UR5e",                # use Sawyer robot
            gripper_types="Robotiq85Gripper",
            use_camera_obs=False,                # do not use pixel observations
            has_offscreen_renderer=False,   # not needed since not using pixel obs
            has_renderer=True,              # make sure we can render to the screen
            reward_shaping=True,            # use dense rewards
            control_freq=500,                # control should happen fast enough so that simulation looks smooth
            horizon=5000
        )
    )

    env = DummyVecEnv([lambda: env])
    # env = Monitor(env, log_dir)
    # Create the callback: check every 1000 steps
    # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    policy_kwargs = dict(activation_fn=tf.nn.LeakyReLU, net_arch=list(eval(args.net_arch)))
    model = PPO('MlpPolicy', env, verbose=2, policy_kwargs=policy_kwargs, n_steps=int(args.n_steps))
    # model = PPO.load("Compliance_32_32_32_new", env=env, tensorboard_log="./PPO2_ur5_Compliance2/",
    #                   n_steps=10, nminibatches=1,learning_rate=10e-6)
    # for i in range(10):
    #     mean_reward = evaluate(model, env=env, num_steps=10,render=True)
    model.learn(total_timesteps=int(args.total_timesteps))#, callback=callback)
    model.save(str(args.model_name))
    # if plotter:
    #
    #     # results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
    #
    #     def moving_average(values, window):
    #         """
    #         Smooth values by doing a moving average
    #         :param values: (numpy array)
    #         :param window: (int)
    #         :return: (numpy array)
    #         """
    #         weights = np.repeat(1.0, window) / window
    #         return np.convolve(values, weights, 'valid')
    #
    #
    #     def plot_results(log_folder, title='Learning Curve'):
    #         """
    #         plot the results
    #
    #         :param log_folder: (str) the save location of the results to plot
    #         :param title: (str) the title of the task to plot
    #         """
    #         x, y = ts2xy(load_results(log_folder), 'timesteps')
    #         y = moving_average(y, window=50)
    #         # Truncate x
    #         x = x[len(x) - len(y):]
    #
    #         fig = plt.figure(title)
    #         plt.plot(x, y)
    #         plt.xlabel('Number of Timesteps')
    #         plt.ylabel('Rewards')
    #         plt.title(title + " Smoothed")
    #         plt.show()
    #
    #     plot_results(log_dir)

    del model

    # logger.record("train/reward", advantages)

    # env = GymWrapper(suite.make("Ur5Stack", has_renderer=True, reward_shaping=True, control_freq=control_freq,
    #                             action_space_def="Compliance", horizon=horizon, dist_error=5e-3,
    #                             angle_error=0.05))
    # env = DummyVecEnv([lambda: env])
    #
    # policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=[32, 32,32])
    # model = PPO('MlpPolicy', env, verbose=2, tensorboard_log="./PPO2_ur5_Compliance/",
    #              policy_kwargs=policy_kwargs, n_steps=10, learning_rate=5e-5)
    # model.learn(total_timesteps=10000, tb_log_name="add_3")
    # model.save("Compliance_all_2_seed_1_smaller_step")
    # del model
    #
    #
    # #for i in range(10):
    #     #mean_reward = evaluate(model, env=env, num_steps=10,render=True)
    #

