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
from collections import deque
import yaml
import tensorflow as tf
import numpy as np
import vizdoom as vd
from tqdm import trange
from helper import start_game, get_game_params, preprocess, test_agent
from q_network import DoubleQNetwork, update_graph, update_target, TBLogger
from buffer import Buffer

# Decide whether to train a new model or to restore from a checkpoint file
load_model = False
save_model = True

with open('take_cover/take_cover.yml') as config_file:
    config = yaml.load(config_file)

# Rename some of the parameters with with shorter anmes
phase1_len = config['phase1_ratio']*config['epochs']
phase2_len = config['phase2_ratio']*config['epochs']

gamma = config['gamma']
start_epsilon = config['start_epsilon']
end_epsilon = config['end_epsilon']
batch_size = config['batch_size']

model_dir = config['model_dir']

# Specify the game scenario and the screen format/resolution
game = start_game(screen_format=vd.ScreenFormat.BGR24,
                  screen_res=vd.ScreenResolution.RES_640X480,
                  config='basic/basic.cfg',
                  sound=config['enable_sound'],
                  visible=config['window_visible'])

width, height, actions = get_game_params(game, config['downscale_ratio'])

tf.reset_default_graph()

target_net = DoubleQNetwork(network_name='target',
                            learning_rate=config['learning_rate'],
                            height=height,
                            width=width,
                            num_actions=len(actions))
DQN = DoubleQNetwork(network_name='online',
                     learning_rate=config['learning_rate'],
                     height=height,
                     width=width,
                     num_actions=len(actions))

exp_buffer = Buffer(size=config['buffer_size'])
session = tf.Session()
saver = tf.train.Saver(max_to_keep=config['num_ckpts'], reshape=True)
weights = tf.trainable_variables()

update_ops = update_graph(weights)

# Set up Tensorboard logging for the online network's training metrics
logger = TBLogger(DQN.loss, DQN.learn_rate, config['log_dir'])

if load_model:
    print('Loading model from', model_dir)
    tf.train.Saver().restore(session, model_dir)

elif not load_model:
    session.run(tf.global_variables_initializer())

game.init()

epoch_rank = list()

"""Accumulate experiences in the buffer using an epsilon-greedy strategy
with three training phases.
"""
for epoch in range(config['epochs']):
    epoch_rewards = list()
    experience = deque(maxlen=2)

    # Initialize the queue with 4 empty states
    queue = deque([list() for i in range(4)], maxlen=4)

    for step in trange(config['steps_per_epoch'], leave=True):
        state = game.get_state()
        state_buffer = preprocess(state.screen_buffer,
                                  config['downscale_ratio'],
                                  preserve_range=False)

        # Add extra dimensions to concatenate the stacks of frames
        state_buffer = state_buffer.reshape(1, 1, height, width)

        for i in range(4):
            queue[i].append(state_buffer)

        # Pop and concatenate the oldest stack of frames
        phi = queue.popleft()
        phi = np.concatenate(phi, axis=1)

        # Explore the environment by choosing random actions
        # with 100% probability for the first phase of training
        # (also choose a random action if there are less
        # than 4 frames in the current state)
        if epoch < phase1_len or phi.shape[1] < 4:
            action = np.random.randint(len(actions))

        # Increase the probability of greedily choosing an action by a
        # constant amount at each epoch in the second phase
        elif epoch < phase2_len:
            epsilon = start_epsilon - (epoch + 1 - phase1_len)*(start_epsilon - end_epsilon)/(phase2_len - phase1_len)

            if np.random.uniform(0, 1) <= epsilon:
                action = np.random.randint(len(actions))
            else:
                action = DQN.choose_action(session, phi)[0]

        # Select a random action with 10% probability in
        # the final phase of training
        else:
            if np.random.uniform(0, 1) <= end_epsilon:
                action = np.random.randint(len(actions))
            else:
                action = DQN.choose_action(session, phi)[0]

        reward = game.make_action(actions[action],
                                  config['frame_delay'])
        done = game.is_episode_finished()

        # Ignores the first states that don't contain 4 frames
        if phi.shape[1] == 4:
            experience.append(phi)

        # Add experiences to the buffer as pairs of consecutive states
        if len(experience) == 2:
            exp_buffer.add_experience((experience[0],
                                       action,
                                       reward,
                                       experience[1],
                                       done))

            # Pop the oldest state to make room for the next one
            experience.popleft()

        # Replace the state we just popped with a new one
        queue.append(list())

        if done:
            # Add zero arrays to stacks with less than 4 frames
            if phi.shape[1] < 4:
                zero_pad_dim = (1, 4 - phi.shape[1], height, width)
                phi = np.concatenate((phi, np.zeros(zero_pad_dim)),
                                     axis=1)

            # Reuse the previous state if the episode has finished
            experience.append(phi)
            exp_buffer.add_experience((experience[0],
                                       action,
                                       reward,
                                       experience[0],
                                       done))
            experience.popleft()

            epoch_rewards.append(game.get_total_reward())

            # Generate a new random seed for each episode
            # (must be less than 9 digits)
            seed = np.random.randint(999999999)
            game.set_seed(seed)
            game.new_episode()

            experience = deque(maxlen=2)

            # Initialize the queue with 4 empty states
            queue = deque([list() for i in range(4)], maxlen=4)

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

            # Calculate how many episodes have already been played
            episode = epoch*config['steps_per_epoch'] + step
            # Log the training loss and learning rate
            logger.write_log(session, DQN, s1, Q2, episode)

    # Increase the discount factor at each epoch until it reaches 0.99
    if gamma < 0.99:
        gamma = 1-.98*(1-gamma)
    elif gamma >= 0.99:
        gamma = 0.99

    # Decrease the learning rate at each epoch
    DQN.update_lr()
    target_net.update_lr()

    print('Epoch {} Mean Reward: {}'.format(epoch + 1, np.mean(epoch_rewards)))

    # Update the target network after every epoch
    update_target(update_ops, session)

    # Save the model and test the agent for 20 episodes every 20 epochs
    if (epoch + 1) % 20 == 0 and epoch > 0:
        if save_model:
            checkpoint = model_dir + '-' + str(epoch + 1)
            print('Epoch {} Model saved to {}'.format(epoch + 1, model_dir))
            saver.save(session, model_dir, global_step=epoch + 1)

        print('Epoch {} test:'.format(epoch + 1))
        test_reward = test_agent(game,
                                 DQN,
                                 num_episodes=20,
                                 config=config,
                                 sound=True,
                                 visible=True,
                                 real_time=True,
                                 session=session,
                                 model_dir=model_dir)
        print('Epoch {} Average Test Reward: {}'.format(epoch + 1,
                                                        test_reward))

        epoch_rank.append((test_reward, epoch + 1))
        game.new_episode()

# Return a sorted list of epoch checkpoints based on
# average test episode reward
print(sorted(epoch_rank, reverse=True))
game.close()
