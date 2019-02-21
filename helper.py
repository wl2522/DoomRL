"""
This module contains helper functions that instantiate new game
instances, downscale game images, and test trained models.
"""
import time
from collections import deque
import numpy as np
import tensorflow as tf
import vizdoom as vd
from skimage.transform import rescale
from skimage.color import rgb2gray
from tqdm import trange


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


def test_agent(game, model, num_episodes, downscale_ratio, delay,
               real_time=True, session=None, model_dir=None):
    """Test the agent using a currently training or previously trained model.
    """
    if model_dir is None:
        sess = tf.Session()
        print('Loading model from', model_dir)
        tf.train.Saver().restore(sess, model_dir)

    # Require an existing session if a pretrained model isn't provided
    else:
        sess = session

    episode_rewards = list()
    width, height, actions = get_game_params(game, downscale_ratio)

    game.init()

    for _ in trange(num_episodes, leave=True):
        # Initialize the queue with 4 empty states
        queue = deque([list() for i in range(4)], maxlen=4)

        # Use a counter to keep track of how many frames have been proccessed
        counter = 0

        game.new_episode()

        while not game.is_episode_finished():
            # Advance the counter first because we check for divisibility by 4
            counter += 1
            # Process only every 4th frame
            if counter % 4 == 0:
                state = game.get_state()
                state_buffer = preprocess(state.screen_buffer,
                                          downscale_ratio,
                                          preserve_range=False)

                # Add extra dimensions to concatenate the stacks of frames
                state_buffer = state_buffer.reshape(1, 1, height, width)

                for i in range(4):
                    queue[i].append(state_buffer)

                # Pop and concatenate the oldest stack of frames
                phi = queue.popleft()
                phi = np.concatenate(phi, axis=1)

                # Add an extra dimension to concatenate the stacks of frames
                phi = np.expand_dims(phi, axis=0)

                # Choose a random action if there are less
                # than 4 frames in the current state
                if phi.shape[1] < 4:
                    action = np.random.randint(len(actions))
                else:
                    action = model.choose_action(sess, phi)[0]

                game.make_action(actions[action], delay)

                # Replace the state we just popped with a new one
                queue.append(list())

                # Delay each time step so that games occur at normal speed
                if real_time:
                    time.sleep(0.02)

        episode_rewards.append(game.get_total_reward())
        print('Test Episode {} Reward: {}'.format(i + 1,
                                                  game.get_total_reward()))

    game.close()

    return np.mean(episode_rewards)
