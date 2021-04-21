import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.controllers import load_controller_config


# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create an environment to visualize on-screen
env = suite.make(
    "Lift",
    # load a Sawyer robot and a Panda robot
    robots=["UR5e"],
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    has_renderer=True,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
)


def get_policy_action(obs):
    # a trained policy could be used here, but we choose a random action
    low, high = env.action_spec
    return np.random.uniform(low, high)


# reset the environment to prepare for a rollout
obs = env.reset()

done = False
ret = 0.
while not done:
    # use observation to decide on an action
    action = get_policy_action(obs)
    obs, reward, done, _ = env.step(action)  # play action
    ret += reward
    env.render()
print("rollout completed with return {}".format(ret))
