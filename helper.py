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
    #
    image = np.moveaxis(image, 0, 2)
    image = np.expand_dims(image, axis=0)
    image = rgb2gray(image)

    if float(downscale_ratio) != 1.0:
        image = rescale(image=image,
                        scale=downscale_ratio,
                        mode='reflect',
                        multichannel=True,
                        preserve_range=preserve_range,
                        anti_aliasing=False)

    # Normalize the image array
    image /= 255.0

    return image


def start_game(screen_format, screen_res, config, sound=False):
    """Start an instance of a game of Doom.

    This function will create a new instance of DoomGame and set
    the paramaters of the game.
    """
    game = vd.DoomGame()
    game.set_screen_format(screen_format)
    game.set_screen_resolution(screen_res)
    game.load_config(config)
    game.set_sound_enabled(sound)

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


def test_agent(game, model, num_episodes, downscale_ratio,
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
            state_buffer = state.screen_buffer
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
