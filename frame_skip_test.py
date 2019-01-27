import numpy as np
import vizdoom as vd
from collections import deque
from helper import start_game, preprocess
from imageio import imwrite

down_sample_ratio = 0.125

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

            state1 = preprocess(state1_buffer, down_sample_ratio)

            for i in range(4):
                queue[i].append(state1//4)

            reward = game.make_action(actions[np.random.randint(len(actions))],
                                      12)

            phi = queue.popleft()

            # Add states to the buffer once there are 4 frames in the oldest one
            if len(phi) == 4:
                experience.append((counter//4, phi))

            # Add experiences  to the buffer as pairs of consecutive states
            if len(experience) == 2:
                buffer.append((experience[0], experience[1]))
                # Pop the first state in the queue to make room for the next state
                experience.popleft()

            # Replace the state we just popped with a new one
            queue.append(list())

            done = game.is_episode_finished()

            # Reuse the previous state if the episode has finished
            if done:
                experience.append((counter//4, phi))
                buffer.append((experience[0], experience[1]))

game.close()

for pair in range(len(buffer)):
    print(buffer[pair][0][0], buffer[pair][1][0])

for stack in range(len(buffer)):
    for state in range(2):
        for image in range(4):
            imwrite('{}_{}_{}.jpg'.format(stack, state, image),
                    buffer[stack][state][1][image].reshape((60, 80, 3)))

