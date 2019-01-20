import yaml
import tensorflow as tf
import numpy as np
import vizdoom as vd
from tqdm import trange

# Decide whether to train a new model or to restore from a checkpoint file
load_model = False
save_model = True

with open('take_cover.yml') as config_file:
    config = yaml.load(config_file)

# Specify the game scenario and the screen format/resolution
game = vd.DoomGame()
game.set_screen_format(vd.ScreenFormat.BGR24)
game.set_screen_resolution(vd.ScreenResolution.RES_640X480)
game.set_depth_buffer_enabled(config['enable_depth_buffer'])
game.load_config('take_cover.cfg')

down_sample_ratio = config['down_sample_ratio']
width = int(game.get_screen_width()*down_sample_ratio)
height = int(game.get_screen_height()*down_sample_ratio)

# Add an extra channel to accomodate the depth buffer if it's enabled
channels = game.get_screen_channels() + int(game.is_depth_buffer_enabled())

# Specify the available actions in the scenario
actions = game.get_available_buttons()
actions = [list(ohe) for ohe in list(np.identity(len(actions)))]

# Define the Q-network learning parameters
frame_delay = config['frame_delay']
buffer_size = config['buffer_size']

epochs = config['epochs']
phase1_len = config['phase1_ratio']*epochs
phase2_len = config['phase2_ratio']*epochs

steps_per_epoch = config['steps_per_epoch']
learning_rate = learning_rate = config['learning_rate']
gamma = config['gamma']
start_epsilon = config['start_epsilon']
end_epsilon = config['end_epsilon']
batch_size = config['batch_size']

model_dir = config['model_dir']
num_ckpts = config['num_ckpts']
