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


def training_iter(game, actions, buffer, steps, stack_len, frame_skip):
    """This function demonstrates the frame skipping algorithm within
    each time step of a game instance.
    """
    experience = deque(maxlen=2)

    # Initialize the queue with empty stacks
    queue = deque([list() for i in range(stack_len)], maxlen=stack_len)

    for step in range(steps):
        state = game.get_state()
        state_buffer = np.moveaxis(state.screen_buffer, 0, 2)

        for i in range(stack_len):
            queue[i].append(state_buffer)

        # Pop and concatenate the oldest stack of frames
        phi = queue.popleft()

        action = np.random.randint(len(actions))
        print(step, action)

        reward = game.make_action(actions[action], frame_skip)
        done = game.is_episode_finished()

        # Ignores the first states that don't contain enough frames
        if len(phi) == stack_len:
            experience.append(phi)

        # Add experiences to the buffer as pairs of consecutive states
        if len(experience) == 2:
            buffer.append((experience[0],
                           action,
                           reward,
                           experience[1],
                           done))

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

            # Initialize the queue with  empty stacks
            queue = deque([list() for i in range(stack_len)], maxlen=stack_len)

        # Add a delay between each time step to slow down the gameplay
        time.sleep(0.01)


def main(steps, stack_len, frame_skip, config_file,
         downscale_ratio=0.125, save=True):
    """Iterate through a game instance and store frames from the game
    into the buffer according to the frame skipping algorithm.
    If save = True, then write images to disk from the buffer
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

    training_iter(game, actions, buffer, steps, stack_len, frame_skip)

    game.close()

    # Save each image from the buffer in the order they were inserted in
    if save:
        for idx, stack in enumerate(buffer):
            # Read each time step's before and after game state frames
            for state in (0, 3):
                for image in range(stack_len):
                    frame = np.squeeze(stack[state][image])
                    # Label each frame to indicate their relative orders
                    order = int(state == 3)
                    imwrite('{}_{}_{}.jpg'.format(idx, order, image),
                            frame)


if __name__ == '__main__':
    main(100, 1, 4, 'take_cover/take_cover.cfg')
