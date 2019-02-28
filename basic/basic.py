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
from tqdm import trange
from q_network import QNetwork, update_graph, update_target, TBLogger, epsilon_greedy
from buffer import Buffer, FrameQueue
from helper import (start_game, get_game_params, preprocess, start_new_episode,
                    test_agent)

# Decide whether to train a new model or to restore from a checkpoint file
load_model = False
save_model = False

with open('basic/basic.yml') as config_file:
    config = yaml.load(config_file)

# Rename some of the parameters with with shorter anmes
phase1_len = config['phase1_ratio']*config['epochs']
phase2_len = config['phase2_ratio']*config['epochs']

gamma = config['gamma']
epsilon_range = config['epsilon_range']
batch_size = config['batch_size']

model_dir = config['model_dir']

# Set to 4 to perform frame stacking or to 1 to train on individual frames
stack_len = config['frame_stack_len']

# Specify the game scenario and the screen format/resolution
game = start_game(screen_format=vd.ScreenFormat.BGR24,
                  screen_res=vd.ScreenResolution.RES_640X480,
                  config='basic/basic.cfg',
                  sound=config['enable_sound'],
                  visible=config['window_visible'])

width, height, actions = get_game_params(game, config['downscale_ratio'])

tf.reset_default_graph()

target_net = QNetwork(name='target',
                      learning_rate=config['learning_rate'],
                      height=height,
                      width=width,
                      num_actions=len(actions),
                      stack_len=stack_len)
DQN = QNetwork(name='online',
               learning_rate=config['learning_rate'],
               height=height,
               width=width,
               num_actions=len(actions),
               stack_len=stack_len)

exp_buffer = Buffer(size=config['buffer_size'])
session = tf.Session()
saver = tf.train.Saver(max_to_keep=config['num_ckpts'], reshape=True)

update_ops = update_graph('online', 'target')

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
    queue = FrameQueue(stack_len)

    for step in trange(config['steps_per_epoch'], leave=True):
        state = game.get_state()
        state_buffer = preprocess(state.screen_buffer,
                                  config['downscale_ratio'],
                                  preserve_range=False)

        frame = queue.stack_frame(state_buffer)

        # Decide whether the next action will be random or not
        choose_random = epsilon_greedy(epoch,
                                       frame,
                                       stack_len,
                                       phase_lens=(phase1_len, phase2_len),
                                       epsilon_range=epsilon_range)
        if choose_random:
            action = np.random.randint(len(actions))
        else:
            action = DQN.choose_action(session, frame)[0]

        reward = game.make_action(actions[action],
                                  config['frame_delay'])
        done = game.is_episode_finished()

        queue.queue_experience(frame, done)
        queue.add_to_buffer(exp_buffer, action, reward, done)

        if done:
            epoch_rewards.append(game.get_total_reward())
            start_new_episode(game)

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

    print('Epoch {} Mean Reward: {}'.format(epoch + 1, np.mean(epoch_rewards)))

    # Update the target network after every epoch
    update_target(update_ops, session)

    # Save the model and test the agent for 20 episodes every 20 epochs
    if (epoch + 1) % config['epochs'] == 0 and epoch > 0:
        if save_model:
            checkpoint = model_dir + '-' + str(epoch + 1)
            print('Epoch {} Model saved to {}'.format(epoch + 1, model_dir))
            saver.save(session, model_dir, global_step=epoch + 1)

        print('Epoch {} test:'.format(epoch + 1))
        test_reward = test_agent(game,
                                 DQN,
                                 num_episodes=20,
                                 config=config,
                                 stack_len=stack_len,
                                 sound=False,
                                 visible=False,
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
