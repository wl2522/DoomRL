import time
import numpy as np
import tensorflow as tf
import vizdoom as vd
from skimage.transform import rescale


def preprocess(image, down_sample_ratio=1):
    """Downsample and normalize an image array representing
    the game state at a given time stamp.
    """
    if float(down_sample_ratio) != 1.0:
        image = rescale(image=image,
                        scale=(down_sample_ratio,
                               down_sample_ratio),
                        mode='reflect')
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    return image


def start_game(screen_format, screen_res, config, enable_depth=False):
    """Start an instance of a game of Doom.

    This function will create a new instance of DoomGame and set
    the paramaters of the game.
    """
    game = vd.DoomGame()
    game.set_screen_format(screen_format)
    game.set_screen_resolution(screen_res)
    game.set_depth_buffer_enabled(enable_depth)
    game.load_config(config)

    return game


def get_game_params(game, down_ratio):
    """
    Get additional game parameters from an instance of a game of Doom.
    """
    width = int(game.get_screen_width()*down_ratio)
    height = int(game.get_screen_height()*down_ratio)

    # Add an extra channel to accomodate the depth buffer if it's enabled
    channels = game.get_screen_channels() + int(game.is_depth_buffer_enabled())

    # Specify the available actions in the scenario
    actions = game.get_available_buttons()

    # Create a list of one hot encoded lists to represent each possible action
    actions = [list(ohe) for ohe in list(np.identity(len(actions)))]

    return width, height, channels, actions


# Test the agent using a currently training or previously trained model

def test_agent(game, model, num_episodes, load_model, depth,
               training=True, session=None, model_dir=None):
    if load_model is True:
        sess = tf.Session()
        print('Loading model from', model_dir)
        tf.train.Saver().restore(sess, model_dir)

# Require an existing session if a pretrained model isn't provided

    elif load_model is False:
        sess = session

    game.set_sound_enabled(False)
    episode_rewards = list()

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

            state1 = preprocess(state_buffer, down_sample_ratio)
            action = model.choose_action(sess, state1)[0]
            reward = game.make_action(actions[action])

# Add a delay between each time step so that the episodes occur at normal speed

            time.sleep(0.02)

        episode_rewards.append(game.get_total_reward())
        print('Test Episode {} Reward: {}'.format(i + 1,
                                                  game.get_total_reward()))

    game.close()

    return np.mean(episode_rewards)
