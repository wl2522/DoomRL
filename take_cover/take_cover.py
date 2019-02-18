"""
Training script for the take_cover.wad scenario.

For each time step, collect the following data:
    1. The current game state
    2. The action that was taken taken
    3. The reward obtained from the chosen action
    4. The next game state
        (store the first game state if the previous action ends the episode)
    5. A variable indicating whether the episode is over yet
"""
import yaml
import tensorflow as tf
import numpy as np
import vizdoom as vd
from collections import deque
from tqdm import trange
from helper import start_game, get_game_params, preprocess, test_agent
from q_network import QNetwork, update_graph, update_target
from buffer import Buffer

# Decide whether to train a new model or to restore from a checkpoint file
load_model = False
save_model = True

with open('take_cover/take_cover.yml') as config_file:
    config = yaml.load(config_file)

# Specify the game scenario and the screen format/resolution
game = start_game(screen_format=vd.ScreenFormat.BGR24,
                  screen_res=vd.ScreenResolution.RES_640X480,
                  config='take_cover/take_cover.cfg',
                  depth=config['enable_depth_buffer'],
                  sound=config['enable_sound'])

width, height, channels, actions = get_game_params(game,
                                                   config['downscale_ratio'])

# Define the Q-network learning parameters
frame_delay = config['frame_delay']
buffer_size = config['buffer_size']

epochs = config['epochs']
phase1_len = config['phase1_ratio']*epochs
phase2_len = config['phase2_ratio']*epochs

steps_per_epoch = config['steps_per_epoch']
learning_rate = config['learning_rate']
gamma = config['gamma']
start_epsilon = config['start_epsilon']
end_epsilon = config['end_epsilon']
batch_size = config['batch_size']

model_dir = config['model_dir']
num_ckpts = config['num_ckpts']

tf.reset_default_graph()

# Instantiate the target network before the online network
# (so that it's updated correctly)
target_net = QNetwork(network_name='target',
                      learning_rate=learning_rate,
                      height=height,
                      width=width,
                      channels=channels,
                      num_actions=len(actions))
DQN = QNetwork(network_name='online',
               learning_rate=learning_rate,
               height=height,
               width=width,
               channels=channels,
               num_actions=len(actions))

exp_buffer = Buffer(size=buffer_size)
session = tf.Session()
saver = tf.train.Saver(max_to_keep=num_ckpts, reshape=True)
weights = tf.trainable_variables()

update_ops = update_graph(weights)

if load_model:
    print('Loading model from', model_dir)
    tf.train.Saver().restore(session, model_dir)

elif not load_model:
    session.run(tf.global_variables_initializer())

#game.init()

epoch_rank = list()

"""Accumulate experiences in the buffer using an epsilon-greedy strategy
with three training phases.
"""
for epoch in range(epochs):
    epoch_rewards = list()
    game.init()

    for step in trange(steps_per_epoch, leave=True):
        experience = deque(maxlen=2)

        # Initialize the queue with 4 empty states
        queue = deque([list() for i in range(4)], maxlen=4)

        # Use a counter to keep track of how many frames have been proccessed
        counter = 0

        while not game.is_episode_finished():
            # Advance the counter first because we check for divisibility by 4
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

                state1 = preprocess(state1_buffer, config['downscale_ratio'])

                for i in range(4):
                    queue[i].append(state1//4)

                # Explore the environment by choosing random actions
                # with 100% probability for the first phase of training
                if epoch < phase1_len:
                    action = np.random.randint(len(actions))

                # Increase the probability of greedily choosing an action by a
                # constant amount at each epoch in the second phase
                elif epoch < phase2_len:
                    epsilon = start_epsilon - (epoch + 1 - phase1_len)*(start_epsilon - end_epsilon)/(phase2_len - phase1_len)

                    if np.random.uniform(0, 1) <= epsilon:
                        action = np.random.randint(len(actions))
                    else:
                        action = DQN.choose_action(session, state1)[0]

                # Select a random action with 10% probability in
                # the final phase of training
                else:
                    if np.random.uniform(0, 1) <= end_epsilon:
                        action = np.random.randint(len(actions))
                    else:
                        action = DQN.choose_action(session, state1)[0]

                reward = game.make_action(actions[action], frame_delay)
                done = game.is_episode_finished()

                if not done:
                    state = game.get_state()

                    if not game.is_depth_buffer_enabled():
                        state2_buffer = np.moveaxis(state.screen_buffer, 0, 2)
                    else:
                        depth_buffer = state.depth_buffer
                        state2_buffer = np.stack((state.screen_buffer,
                                                  depth_buffer), axis=-1)

                    state2 = preprocess(state2_buffer,
                                        config['downscale_ratio'])

                elif done:
                    state2 = state1

                # Add the experience obtained from each time step to the buffer
                exp_buffer.add_experience((state1,
                                           action,
                                           reward,
                                           state2,
                                           done))

        # Sample a minibatch from the buffer
        # (if there are enough experiences that have been saved already)
        if exp_buffer.length > batch_size:
            s1, a, r, s2, terminal = exp_buffer.sample_buffer(batch_size)

            # Get the target values from the target Q-network
            target_Q = np.max(target_net.get_Q_values(session, s2), axis=1)

            # Train the online Q-network by using a minibatch to
            # update the action-value function
            Q2 = DQN.get_Q_values(session, s1)
            Q2[np.arange(batch_size), a] = r + gamma*(1 - terminal)*target_Q
            DQN.calculate_loss(session, s1, Q2)

        epoch_rewards.append(game.get_total_reward())

    game.close()

    # Increase the discount factor at each epoch until it reaches 0.99
    if gamma < 0.99:
        gamma = 1-.98*(1-gamma)
    elif gamma >= 0.99:
        gamma = 0.99

    # Decrease the learning rate at each epoch
    DQN.update_lr()
    target_net.update_lr()

    print('Epoch {} Mean Reward: {}'.format(epoch + 1, np.mean(epoch_rewards)))

    # Update the target network every 10 epochs
    if (epoch + 1) % 10 == 0 and epoch > 0:
        update_target(update_ops, session)

    # Save the model and test the agent for 10 episodes every 20 epochs
    if (epoch + 1) % 10 == 0 and epoch > 0:
        if save_model:
            checkpoint = model_dir + '-' + str(epoch + 1)
            print('Epoch {} Model saved to {}'.format(epoch + 1, model_dir))
            saver.save(session, model_dir, global_step=epoch + 1)

        update_target(update_ops, session)

        print('Epoch {} test:'.format(epoch + 1))
        test_reward = test_agent(game,
                                 DQN,
                                 num_episodes=20,
                                 load_model=False,
                                 depth=False,
                                 session=session,
                                 model_dir=model_dir)
        print('Epoch {} Average Test Reward: {}'.format(epoch + 1,
                                                        test_reward))

        epoch_rank.append((test_reward, epoch + 1))

# Return a sorted list of epoch checkpoints based on
# average test episode reward
print(sorted(epoch_rank, reverse=True))
#game.close()
