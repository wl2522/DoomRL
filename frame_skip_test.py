"""
This script adapts the frame skipping algorithm demonstrated
in frame_skip.py and uses it on image frames obtained
during an actual game instance of Doom.

Adapted from: https://github.com/mwydmuch/ViZDoom/issues/296
"""
import time
from collections import deque
import numpy as np
from imageio import imwrite
import vizdoom as vd
from helper import start_game, get_game_params


def training_iter(game, actions, buffer, num_steps):
    """This function demonstrates the frame skipping algorithm within
    each time step of a game instance.
    """
    experience = deque(maxlen=2)

    # Initialize the queue with 4 empty states
    queue = deque([list() for i in range(4)], maxlen=4)

    for step in range(num_steps):
        state = game.get_state()
        state_buffer = np.moveaxis(state.screen_buffer, 0, 2)

        for i in range(4):
            queue[i].append(state_buffer)

        # Pop and concatenate the oldest stack of frames
        phi = queue.popleft()

        action = np.random.randint(len(actions))
        print(step, action)

        reward = game.make_action(actions[action], 4)
        done = game.is_episode_finished()

        # Ignores the first states that don't contain 4 frames
        if len(phi) == 4:
            experience.append(phi)

        # Add experiences to the buffer as pairs of consecutive states
        if len(experience) == 2:
            buffer.append((experience[0],
                           action,
                           reward,
                           experience[1],
                           done))

            # Pop the oldest state to make room for the next one
            experience.popleft()

        # Replace the state we just popped with a new one
        queue.append(list())

        if done:
            # Reuse the previous state if the episode has finished
            experience.append(phi)
            buffer.append((experience[0],
                           action,
                           reward,
                           experience[0],
                           done))
            experience.popleft()
            game.new_episode()

            experience = deque(maxlen=2)

            # Initialize the queue with 4 empty states
            queue = deque([list() for i in range(4)], maxlen=4)

        # Add a delay between each time step to slow down the gameplay
        time.sleep(0.01)


def main(num_steps, config_file, downscale_ratio=0.125, save_images=True):
    """Iterate through a game instance and store frames from the game
    into the buffer according to the frame skipping algorithm.
    If save_images = True then write images from the buffer to disk
    at the end of the script to confirm correct results
    """
    game = start_game(screen_format=vd.ScreenFormat.BGR24,
                      screen_res=vd.ScreenResolution.RES_640X480,
                      config=config_file,
                      sound=False,
                      visible=True)
    _, _, actions = get_game_params(game, downscale_ratio)
    buffer = list()

    game.init()

    training_iter(game, actions, buffer, num_steps)

    game.close()

    # Save each image from the buffer in the order they were inserted in
    if save_images:
        for idx, stack in enumerate(buffer):
            # Read each time step's before and after game state frames
            for state in (0, 3):
                for image in range(4):
                    frame = np.squeeze(stack[state][image])
                    # Label each frame to indicate their relative orders
                    order = int(state == 3)
                    imwrite('{}_{}_{}.jpg'.format(idx, order, image),
                            frame)


if __name__ == '__main__':
    main(100, 'take_cover/take_cover.cfg')
