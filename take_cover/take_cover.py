"""
For each time step, collect the following data:
    1. The current game state
    2. The action that was taken taken
    3. The reward obtained from the chosen action
    4. The next game state
        (store the first game state if the previous action ends the episode)
    5. A variable indicating whether the episode is over yet
"""

import yaml
import tensorflow as tf
import numpy as np
import vizdoom as vd
from tqdm import trange
from helper import start_game, get_game_params, preprocess, test_agent
from q_network import QNetwork, update_graph, update_target
from buffer import Buffer

# Decide whether to train a new model or to restore from a checkpoint file
load_model = False
save_model = True

with open('take_cover.yml') as config_file:
    config = yaml.load(config_file)

# Specify the game scenario and the screen format/resolution
game = start_game(screen_format=vd.ScreenFormat.BGR24,
                  screen_res=vd.ScreenResolution.RES_640X480,
                  config='take_cover.cfg',
                  depth=config['enable_depth_buffer'],
                  sound=config['enable_sound'])

width, height, channels, actions = get_game_params(game,
                                                   config['down_sample_ratio'])

# Define the Q-network learning parameters
frame_delay = config['frame_delay']
buffer_size = config['buffer_size']

epochs = config['epochs']
phase1_len = config['phase1_ratio']*epochs
phase2_len = config['phase2_ratio']*epochs

steps_per_epoch = config['steps_per_epoch']
learning_rate = config['learning_rate']
gamma = config['gamma']
start_epsilon = config['start_epsilon']
end_epsilon = config['end_epsilon']
batch_size = config['batch_size']

model_dir = config['model_dir']
num_ckpts = config['num_ckpts']
