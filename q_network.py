import tensorflow as tf
import numpy as np


class BaseNetwork:
    """A base class that has attributes and functions that are needed by
    all of the Deep Q-networks.
    """
    def __init__(self, name, height, width, num_actions, stack_len,
                 learning_rate):
        self.name = name
        self.height = height
        self.width = width
        self.stack_len = stack_len
        self.num_actions = num_actions

        # Map the network to its network name in the Tensorflow graph
        with tf.variable_scope(name):
            self.learn_rate = learning_rate
            self.s_t = tf.placeholder(tf.float32,
                                      shape=[None,
                                             stack_len,
                                             self.height,
                                             self.width],
                                      name=self.name + '_state'
                                      )
            self.a_t = tf.placeholder(tf.int32,
                                      shape=[None],
                                      name=self.name + '_action'
                                      )
            self.Q_target = tf.placeholder(tf.float32,
                                           shape=[None, self.num_actions],
                                           name=self.name + '_Q_target'
                                           )

    def update_lr(self):
        """Reduce the learning rate of the Q-Network by 2%.
        """
        self.learn_rate = 0.98*self.learn_rate

        return self.learn_rate


class QNetwork(BaseNetwork):
    """Create a Q-network to estimate expected values and choose
    actions for a given state.
    """
    def __init__(self, name, height, width, num_actions, stack_len=1,
                 learning_rate=0.001):
        super().__init__(name,
                         height,
                         width,
                         num_actions,
                         stack_len,
                         learning_rate)

        # Map the network to its network name in the Tensorflow graph
        with tf.variable_scope(self.name):
            self.conv1 = tf.layers.conv2d(inputs=self.s_t,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding='valid',
                                          data_format='channels_first',
                                          activation=tf.nn.relu,
                                          name=self.name + '_conv1_layer'
                                          )
            self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding='valid',
                                          activation=tf.nn.relu,
                                          name=self.name + '_conv2_layer'
                                          )
            self.flatten = tf.layers.flatten(inputs=self.conv2,
                                             name=self.name + '_flatten'
                                             )
            self.dense = tf.layers.dense(inputs=self.flatten,
                                         units=512,
                                         activation=tf.nn.relu,
                                         name=self.name + '_dense1_layer'
                                         )
            self.Q_values = tf.layers.dense(inputs=self.dense,
                                            units=self.num_actions,
                                            activation=None,
                                            name=self.name + '_output_layer'
                                            )

            self.best_action = tf.argmax(self.Q_values, 1)
            self.loss = tf.losses.mean_squared_error(self.Q_values,
                                                     self.Q_target)
            self.adam = tf.train.AdamOptimizer(learning_rate=self.learn_rate,
                                               name=self.name + '_adam'
                                               )
            self.train = self.adam.minimize(self.loss)

    def calculate_loss(self, session, s, q):
        """Compute the mean squared error for state s and apply the gradients.
        """
        L, _ = session.run([self.loss, self.train],
                           feed_dict={self.s_t: s,
                                      self.Q_target: q})

        return L

    def get_Q_values(self, session, s):
        """Return the array of Q-values associated with a given state.
        """
        Q = session.run(self.Q_values,
                        feed_dict={self.s_t: s})

        return Q

    def choose_action(self, session, s):
        """Return the best action based on the a given state.
        """
        a = session.run(self.best_action,
                        feed_dict={self.s_t: s})

        return a


class DoubleQNetwork(BaseNetwork):
    """Create a deep Q-network with 3 convolutional layers. Two of these
    should be instantiated: one to estimate expected values and one to
    choose actions for a given state.
    """
    def __init__(self, name, height, width, num_actions, stack_len=4,
                 learning_rate=0.001):
        super().__init__(name,
                         height,
                         width,
                         num_actions,
                         stack_len,
                         learning_rate)

        # Map the network to its network name in the Tensorflow graph
        with tf.variable_scope(self.name):
            self.conv1 = tf.layers.conv2d(inputs=self.s_t,
                                          filters=32,
                                          kernel_size=[6, 6],
                                          strides=[3, 3],
                                          padding='valid',
                                          data_format='channels_first',
                                          activation=tf.nn.relu,
                                          name=self.name + '_conv1_layer'
                                          )
            self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding='valid',
                                          activation=tf.nn.relu,
                                          name=self.name + '_conv2_layer'
                                          )
            self.conv3 = tf.layers.conv2d(inputs=self.conv2,
                                          filters=128,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding='valid',
                                          activation=tf.nn.relu,
                                          name=self.name + 'conv3_layer'
                                          )
            self.flatten = tf.layers.flatten(inputs=self.conv3,
                                             name=self.name + '_flatten'
                                             )
            self.dense = tf.layers.dense(inputs=self.flatten,
                                         units=512,
                                         activation=tf.nn.relu,
                                         name=self.name + '_dense1_layer'
                                         )
            self.Q_values = tf.layers.dense(inputs=self.dense,
                                            units=self.num_actions,
                                            activation=None,
                                            name=self.name + '_output_layer'
                                            )

            self.best_action = tf.argmax(self.Q_values, 1)
            self.loss = tf.losses.mean_squared_error(self.Q_values,
                                                     self.Q_target)
            self.adam = tf.train.AdamOptimizer(learning_rate=self.learn_rate,
                                               name=self.name + '_adam'
                                               )
            self.train = self.adam.minimize(self.loss)

    def calculate_loss(self, session, s, q):
        """Compute the mean squared error for state s and apply the gradients.
        """
        L, _ = session.run([self.loss, self.train],
                           feed_dict={self.s_t: s,
                                      self.Q_target: q})

        return L

    def get_Q_values(self, session, s):
        """Return the array of Q-values associated with a given state.
        """
        Q = session.run(self.Q_values,
                        feed_dict={self.s_t: s})

        return Q

    def choose_action(self, session, s):
        """Return the best action based on the a given state.
        """
        a = session.run(self.best_action,
                        feed_dict={self.s_t: s})

        return a


def epsilon_greedy(epoch, frame, stack_len, phase_lens, epsilon_range):
    """Implement an epsilon-greedy strategy that decides whether
    an agent performs a random action (with probability epsilon) or
    choose an action greedily.
    """
    def epsilon_decay(epoch, phase_lens, epsilon_range):
        """Update the value of epsilon based on the current epoch number,
        the provided range of epsilon values, and the exploration scheduled
        defined by a tuple.
        Epsilon is the parameter that determines the probability of
        the agent performing a random action instead of deciding on
        an action using a greedy strategy.

        At each epoch between the exploration phase and final phase,
        epsilon linearly decreases between the given range of epsilon values.
        """
        # Indicate how many times epsilon has been updated so far
        update_step = epoch + 1 - phase_lens[0]

        # Calculate how much epsilon should decrease by at each update step
        decay_factor = (epsilon_range[0] - epsilon_range[1]) /    \
                       (phase_lens[1] - phase_lens[0])
        epsilon = epsilon_range[0] - update_step*decay_factor

        return epsilon

    epsilon = epsilon_decay(epoch, phase_lens, epsilon_range)

    # Explore the environment by choosing random actions
    # with 100% probability for the first phase of training
    # (also choose a random action if there are less
    # frames than the stack length in the current stack)
    if epoch < phase_lens[0] or frame.shape[1] < stack_len:
        random_action = True

    # Increase the probability of greedily choosing an action by a
    # constant amount at each epoch in the second phase
    elif epoch < phase_lens[1]:
        random_action = bool(np.random.uniform(0, 1) <= epsilon)

    # Select a random action with 10% probability in
    # the final phase of training
    else:
        random_action = bool(np.random.uniform(0, 1) <= epsilon_range[1])

    return random_action


def update_graph(from_network_name, to_network_name):
    """When training a double DQN, create a list of variable
    update operations. These operations assign weight values from
    one network to another.
    """
    # Get the parameters of our first network
    online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    from_network_name)

    # Get the parameters of our second network
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    to_network_name)

    update_ops = list()

    # Update our second network's parameters with first network's
    for online_vars, target_vars in zip(online_vars, target_vars):
        update_ops.append(target_vars.assign(online_vars))
    return update_ops


def update_target(update_ops, session):
    """When training a double DQN consisting of an online and a target
    network, update the target network parameters to match those of
    the online network.
    """
    for op in update_ops:
        session.run(op)


class TBLogger:
    """Create a Tensorboard logger to record training loss and learning rate
    while training the online Q-Network.
    """
    def __init__(self, dqn_loss, dqn_learning_rate, log_dir='./tensorboard'):
        tf.summary.scalar('training_loss', dqn_loss)
        tf.summary.scalar('learning_rate', dqn_learning_rate)

        self.summarize = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(log_dir,
                                            graph=tf.get_default_graph())

    def write_log(self, session, q_network, s, q, train_iter):
        """Update the Tensorboard logs after an iteration of training.
        """
        iter_log = session.run(self.summarize,
                               feed_dict={q_network.s_t: s,
                                          q_network.Q_target: q})
        self.writer.add_summary(iter_log, train_iter)
        self.writer.flush()
