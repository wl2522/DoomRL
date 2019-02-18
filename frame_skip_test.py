"""
This script adapts the frame skipping algorithm demonstrated
in frame_skip.py and uses it on image frames obtained
during an actual game instance of Doom.

Adapted from: https://github.com/mwydmuch/ViZDoom/issues/296
"""
import time
import numpy as np
import vizdoom as vd
from collections import deque
from imageio import imwrite
from helper import start_game, get_game_params


def training_iter(game, actions, buffer):
    """This function demonstrates the frame skipping algorithm within
    each time step of a game instance.
    """
    experience = deque(maxlen=2)

    # Initialize the frame-skipping algorithm with a queue of 4 empty states
    queue = deque([list() for i in range(4)], maxlen=4)

    # Use a counter to keep track of how many frames have been proccessed
    counter = 0

    while not game.is_episode_finished():
        # Increment the counter first because we check for divisibility by 4
        counter += 1
        # Process only every 4th frame
        if counter % 4 == 0:
            state = game.get_state()

            if not game.is_depth_buffer_enabled():
                state1_buffer = np.moveaxis(state.screen_buffer, 0, 2)
            else:
                depth_buffer = np.expand_dims(state.depth_buffer, 0)
                state1_buffer = np.stack((state.screen_buffer,
                                          depth_buffer), axis=-1)

            for i in range(4):
                queue[i].append(state1_buffer//4)

            action = actions[np.random.randint(len(actions))]
            print(counter, action)
            reward = game.make_action(action, 4)
            done = game.is_episode_finished()

            phi = queue.popleft()

            # Ignores the first states that don't contain 4 frames
            if len(phi) == 4:
                experience.append((counter//4, phi))

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

            # Reuse the previous state if the episode has finished
            if done:
                experience.append((counter//4, phi))
                buffer.append((experience[0],
                               action,
                               reward,
                               experience[1],
                               done))

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
                      config=config_file)
    _, _, _, actions = get_game_params(game, downscale_ratio)
    buffer = list()

    game.init()

    for _ in range(num_steps):
        training_iter(game, actions, buffer)

    game.close()

    # Save each image from the buffer in the order they were inserted in
    if save_images:
        for idx, stack in enumerate(buffer):
            # Read the before and after frames in each game state
            for state in (0, 3):
                for image in range(4):
                    # Scale the image array to 0-255 to increase brightness
                    frame = np.squeeze(stack[state][1][image])*4
                    # Label each state's before frame as 0 and after frame as 1
                    order = int(state == 3)
                    imwrite('{}_{}_{}.jpg'.format(idx, order, image),
                            frame)


if __name__ == '__main__':
    main(16, 'take_cover/take_cover.cfg')
