from my_environment import MyEnv
import numpy as np

if __name__ == "__main__":
	# create environment instance
	env = MyEnv(robots="UR5e",
				has_renderer=True,
				has_offscreen_renderer=False,
				use_camera_obs=False,)

	# reset the environment
	env.reset()

	for i in range(10000):
		action = np.random.randn(env.robots[0].dof) # sample random action
		obs, reward, done, info = env.step(action)  # take action in the environment
		env.render()  # render on display