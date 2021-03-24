import gym

# from stable_baselines3.common.policies import
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from robosuite.wrappers import GymWrapper
import robosuite as suite
import numpy as np
import time
import torch as tf
from matplotlib import pyplot as plt
import csv

def evaluate(model,env, num_steps=1000,render=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    rewards_all=[]
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        file = open("testfile2.txt", "w")

        file.write(str(action))

        file.close()
        if i==0:
            action2=action
        #[x, y, z]=obs[0,6:9]
        #[t_x, t_y, t_z]=obs[0,9:12]

        #right_K_pos = np.identity(3)*0
        #right_K_ori = np.identity(3)*0
        #right_C_pos = np.identity(3)*0
        #right_C_ori = np.identity(3)*0

        #param=[x, y, z, t_x, t_y, t_z,
         #right_K_pos[0, 0], right_K_pos[1, 1], right_K_pos[2, 2],
         #right_K_ori[0, 0], right_K_ori[1, 1], right_K_ori[2, 2],
         #right_C_pos[0, 0], right_C_pos[1, 1], right_C_pos[2, 2],
         #right_C_ori[0, 0], right_C_ori[1, 1], right_C_ori[2, 2]]

        #param=np.expand_dims(np.array(param),axis=0)

        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)
        print(rewards)
        if render:
            env.render()
            #time.sleep(0.1)
        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)
        rewards_all.append(rewards)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", np.sum(rewards_all), "Num episodes:", len(episode_rewards))

    return mean_100ep_reward

def evaluate_success(model,env,dist_error,angle_error,num_iterations=1000,render=False,):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """

    episode_rewards = [0.0]
    obs = env.reset()
    rewards_all=[]
    count=0
    time_all=[]
    for i in range(num_iterations):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)

        #[x, y, z]=obs[0,6:9]
        #[t_x, t_y, t_z]=obs[0,9:12]

        #right_K_pos = np.identity(3)*0
        #right_K_ori = np.identity(3)*0
        #right_C_pos = np.identity(3)*0
        #right_C_ori = np.identity(3)*0

        #param=[x, y, z, t_x, t_y, t_z,
         #right_K_pos[0, 0], right_K_pos[1, 1], right_K_pos[2, 2],
         #right_K_ori[0, 0], right_K_ori[1, 1], right_K_ori[2, 2],
         #right_C_pos[0, 0], right_C_pos[1, 1], right_C_pos[2, 2],
         #right_C_ori[0, 0], right_C_ori[1, 1], right_C_ori[2, 2]]

        #param=np.expand_dims(np.array(param),axis=0)

        # here, action, rewards and dones are arrays
        # because we are using vectorized env

        obs, rewards, dones, info = env.step(action)
        print(rewards)
        if dones[0]:
            if rewards==35000.0:
                count+=1
                time_all.append(info[0].pop('time'))
            obs = env.reset()
        # Stats
    success_rate=(count/num_iterations)*100
    row=[str(dist_error),str(angle_error),str(success_rate),str(np.average(time_all))]
    with open('error_success_round_force_moment_100Hz.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
    #print("Suceess is {} out of {} = to this {}%:".format(count, num_iterations,success_rate))


if __name__ == "__main__":

    horizon=5000
    control_freq=500
    number_of_slices=10
    num_iterations=20


    env = GymWrapper(suite.make("Ur5Stack", has_renderer=True, reward_shaping=True, control_freq=control_freq,
                                action_space_def="Impedance", horizon=horizon, dist_error=5e-3,
                                angle_error=0.05))
    env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(activation_fn=tf.nn.LeakyReLU, net_arch=[32, 32])
    model = PPO('MlpPolicy', env, verbose=2, tensorboard_log="./PPO_ur5_Impedance_shir_wo_reward/",
                policy_kwargs=policy_kwargs, n_steps=10)
    # model = PPO.load("Compliance_32_32_32_new", env=env, tensorboard_log="./PPO2_ur5_Compliance2/",
    #                   n_steps=10, nminibatches=1,learning_rate=10e-6)
    # for i in range(10):
    #     mean_reward = evaluate(model, env=env, num_steps=10,render=True)
    model.learn(total_timesteps=10000, tb_log_name="Impedance_32_32_shir_torch")
    model.save("Impedance_32_32_shir_torch")
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
