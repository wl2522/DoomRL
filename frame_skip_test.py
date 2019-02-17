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


down_sample_ratio = 0.125
config_file = 'take_cover/take_cover.cfg'

# Save images in the buffer at the end of the script to confirm correct results
save_images = True

# Specify the game scenario and the screen format/resolution
game, _, _, _, actions = start_game(screen_format=vd.ScreenFormat.BGR24,
                              screen_res=vd.ScreenResolution.RES_640X480,
                              enable_depth=False,
                              config='take_cover/take_cover.cfg',
                              down_ratio=down_sample_ratio)

buffer = list()

game.init()

for step in range(16):
    queue = deque(maxlen=4)
    experience = deque(maxlen=2)

    # Initialize the frame-skipping algorithm with a queue of 4 empty states
    queue = deque([list() for i in range(4)])
    # Use a counter to keep track of how many frames have been proccessed
    counter = 0

    while not game.is_episode_finished():
        # Increment the counter first so that we can check for divisibility by 4
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

            # Add states to the buffer once there are 4 frames in the oldest one
            if len(phi) == 4:
                experience.append((counter//4, phi))

            # Add experiences to the buffer as pairs of consecutive states
            if len(experience) == 2:
                buffer.append((experience[0],
                               action,
                               reward,
                               experience[1],
                               done))

                # Pop the first state in the queue to make room for the next state
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

game.close()

# Save each image from the buffer in the order they were inserted in
if save_images:
    for idx, stack in enumerate(buffer):
        # Read the before and after frames in each game state
        for state in (0, 3):
            for image in range(4):
                # Convert the image array to 0-255 range to increase brightness
                frame = np.squeeze(stack[state][1][image])*4
                # Label each state's before frame as 0 and after frame as 1
                order = int(state == 3)
                imwrite('{}_{}_{}.jpg'.format(idx, order, image),
                        frame)
