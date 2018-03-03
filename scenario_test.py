# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 23:08:20 2018

@author: Keep Tryin'
"""
#%%

import time
import vizdoom as vd
import numpy as np
from skimage.transform import rescale


def preprocess(image, down_sample_ratio=1):
    if down_sample_ratio != 1:
        image = rescale(image=image, scale=down_sample_ratio)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)

    return image

down_sample_ratio=0.5
#%%


game = vd.DoomGame()
game.set_screen_format(vd.ScreenFormat.BGR24)
game.set_depth_buffer_enabled(False)
game.set_screen_resolution(vd.ScreenResolution.RES_640X480)
game.set_mode(vd.Mode.PLAYER)
game.load_config('./take_cover/take_cover.cfg')

available_actions = game.get_available_buttons()
actions = [list(ohe) for ohe in list(np.identity(len(available_actions)))]
num_actions = len(available_actions)

episodes=1
game.init()



times = list()
for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()
    start = time.time()
    while not game.is_episode_finished():

        # Gets the state
        state = game.get_state()

        # Which consists of:
        n           = state.number
        vars        = state.game_variables
        screen_buf  = state.screen_buffer
        depth_buf   = state.depth_buffer
        labels_buf  = state.labels_buffer
        automap_buf = state.automap_buffer
        labels      = state.labels

        # Makes a random action and get remember reward.
        action = np.random.randint(num_actions)
        r = game.make_action(actions[action])
        time.sleep(0.02)
        # Makes a "prolonged" action and skip frames:
        # skiprate = 4
        # r = game.make_acti,on(choice(actions), skiprate)

        # The same could be achieved with:
        # game.set_action(choice(actions))
        # game.advance_action(skiprate)
        # r = game.get_last_reward()

        # Prints state's game variables and reward.
        print("State #" + str(n), "Tic #", + state.tic)
        print("Game variables:", vars)
        print("Action:", action)
        print("Reward:", r)
        print("=====================")
    times.append(time.time() - start)
    print(times)

    # Check how the episode went.
    print("Episode finished.")
    print("Total reward:", game.get_total_reward())
    print("************************")
game.close()
print(min(times))

