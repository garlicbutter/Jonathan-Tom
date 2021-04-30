import robosuite as suite
from robosuite import load_controller_config

# Load the desired controller's default config as a dict
controller_config = load_controller_config(default_controller='OSC_POSE')
