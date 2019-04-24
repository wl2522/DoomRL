"""
This module contains helper functions that instantiate new game
instances, downscale game images, and test trained models.
"""
import time
import numpy as np
import tensorflow as tf
import vizdoom as vd
from skimage.transform import rescale
from skimage.color import rgb2gray
from buffer import FrameQueue


def preprocess(image, downscale_ratio=1, preserve_range=False):
    """Downsample and preprocess an image array representing
    the game state at a given time stamp. The preprocessing steps are:
        1. Move the color channel dimension to the end position as is
        required by the rescale() function in scikit-image
        2. Add an extra dimension at the front position to stack
        frames on
        3. Downscale the image and normalize the image array so
        that each array value is between 0 and 1
        4. Convert the image to grayscale (remove the color dimension)
        since color adds unnecessary dimensions (and extra complexity)
        to the model training process
    """
    image = np.moveaxis(image, 0, 2)

    if float(downscale_ratio) != 1.0:
        image = rescale(image=image,
                        scale=(downscale_ratio,
                               downscale_ratio),
                        mode='reflect',
                        multichannel=True,
                        preserve_range=preserve_range,
                        anti_aliasing=False)

    image = rgb2gray(image)

    # Add extra dimensions to concatenate the stacks of frames on
    image = image.reshape((1, 1, *image.shape))

    return image


def start_game(screen_format, screen_res, config, sound=False, visible=False):
    """Start an instance of a game of Doom.

    This function will create a new instance of DoomGame and set
    the paramaters of the game.
    """
    game = vd.DoomGame()
    game.set_screen_format(screen_format)
    game.set_screen_resolution(screen_res)
    game.load_config(config)
    game.set_sound_enabled(sound)
    game.set_window_visible(visible)

    return game


def get_game_params(game, downscale_ratio):
    """
    Get additional game parameters from an instance of a game of Doom.
    """
    width = int(game.get_screen_width()*downscale_ratio)
    height = int(game.get_screen_height()*downscale_ratio)

    # Specify the available actions in the scenario
    actions = game.get_available_buttons()

    # Create a list of one hot encoded lists to represent each possible action
    actions = [list(ohe) for ohe in list(np.identity(len(actions)))]

    return width, height, actions


def start_new_episode(game):
    """Randomly generate a number, set it as the
    random seed within the game instance, and start a
    new episode.
    The generated number can be at most 9 digits since trying to
    set a 10 digit random seed will cause an error in ViZDoom.
    """
    seed = np.random.randint(999999999)
    game.set_seed(seed)
    game.new_episode()


def test_agent(game, model, num_episodes, config, stack_len, sound=False,
               visible=True, real_time=True, model_dir=None):
    """Test the agent using a currently training or previously trained model.
    Parameters related to model training and game instance settings
    are read from a dictionary.
    """
    # Initiate a new game if the sound or visible parameters
    # differ from what's in the config dictionary
    if sound != config['enable_sound'] or visible != config['window_visible']:
        game.close()
        game.set_window_visible(visible)
        game.set_sound_enabled(sound)
        game.init()

    episode_rewards = list()
    width, height, actions = get_game_params(game, config['downscale_ratio'])

    for episode in range(num_episodes):
        queue = FrameQueue(stack_len)
        start_new_episode(game)

        while not game.is_episode_finished():
            state = game.get_state()
            state_buffer = preprocess(state.screen_buffer,
                                      config['downscale_ratio'],
                                      preserve_range=False)

            # Add extra dimensions to concatenate the stacks of frames
            state_buffer = state_buffer.reshape(1, 1, height, width)
            phi = queue.stack_frame(state_buffer)

            # Choose a random action if there are less
            # than the required number of frames in the current state
            if phi.shape[1] < stack_len:
                action = np.random.randint(len(actions))
            else:
                action = model.choose_action(phi)[0]

            game.make_action(actions[action], config['frame_delay'])

            # Delay each time step so that games occur at normal speed
            if real_time:
                time.sleep(0.02)

        episode_rewards.append(game.get_total_reward())
        print('Test Episode {} Reward: {}'.format(episode + 1,
                                                  game.get_total_reward()))

    # Create a new game instance with the previous sound/window settings
    if sound != config['enable_sound'] or visible != config['window_visible']:
        game.close()
        game.set_window_visible(config['window_visible'])
        game.set_sound_enabled(config['enable_sound'])
        game.init()

    return np.mean(episode_rewards)
