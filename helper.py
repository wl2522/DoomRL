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


def preprocess(image, downscale_ratio=1, preserve_range=False):
    """Downsample and normalize an image array representing
    the game state at a given time stamp. The final image is
    converted to grayscale since color is unnecessary for training.
    """
    if float(downscale_ratio) != 1.0:
        image = rescale(image=image,
                        scale=(downscale_ratio,
                               downscale_ratio),
                        mode='reflect',
                        multichannel=True,
                        preserve_range=preserve_range,
                        anti_aliasing=False)
    image = rgb2gray(image)
    image = image.astype(np.float32)

    # Normalize the image array
    image /= 255.0
    image = np.expand_dims(image, axis=0)

    return image


def start_game(screen_format, screen_res, config, depth=False, sound=False):
    """Start an instance of a game of Doom.

    This function will create a new instance of DoomGame and set
    the paramaters of the game.
    """
    game = vd.DoomGame()
    game.set_screen_format(screen_format)
    game.set_screen_resolution(screen_res)
    game.set_depth_buffer_enabled(depth)
    game.load_config(config)
    game.set_sound_enabled(sound)

    return game


def get_game_params(game, downscale_ratio):
    """
    Get additional game parameters from an instance of a game of Doom.
    """
    width = int(game.get_screen_width()*downscale_ratio)
    height = int(game.get_screen_height()*downscale_ratio)

    # Add an extra channel to accomodate the depth buffer if it's enabled
    channels = game.get_screen_channels() + int(game.is_depth_buffer_enabled())

    # Specify the available actions in the scenario
    actions = game.get_available_buttons()

    # Create a list of one hot encoded lists to represent each possible action
    actions = [list(ohe) for ohe in list(np.identity(len(actions)))]

    return width, height, channels, actions


def test_agent(game, model, num_episodes, depth, downscale_ratio,
               session=None, load_model=False, model_dir=None):
    """Test the agent using a currently training or previously trained model.
    """
    if load_model:
        sess = tf.Session()
        print('Loading model from', model_dir)
        tf.train.Saver().restore(sess, model_dir)

    # Require an existing session if a pretrained model isn't provided
    elif not load_model:
        sess = session

    episode_rewards = list()
    _, _, _, actions = get_game_params(game, downscale_ratio)

    game.init()

    for i in range(num_episodes):
        game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state()

            if depth is False:
                state_buffer = np.moveaxis(state.screen_buffer, 0, 2)
            elif depth is True:
                depth_buffer = state.depth_buffer
                state_buffer = np.stack((state.screen_buffer,
                                         depth_buffer), axis=-1)

            state1 = preprocess(state_buffer, downscale_ratio)
            action = model.choose_action(sess, state1)[0]
            game.make_action(actions[action])

            # Add a delay at each time step so that games occur at normal speed
            time.sleep(0.02)

        episode_rewards.append(game.get_total_reward())
        print('Test Episode {} Reward: {}'.format(i + 1,
                                                  game.get_total_reward()))

    game.close()

    return np.mean(episode_rewards)
