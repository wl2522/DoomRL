import tensorflow as tf


class QNetwork:
    """Create a Q-network to estimate expected values and choose
    actions for a given state.
    """
    def __init__(self, network_name, height, width, num_actions,
                 learning_rate=0.001):
        # Map the network to its network name in the Tensorflow graph
        with tf.variable_scope(network_name):
            self.learning_rate = learning_rate
            self.s_t = tf.placeholder(tf.float32,
                                      shape=[None, 4, height, width],
                                      name=network_name + '_state'
                                      )
            self.a_t = tf.placeholder(tf.int32,
                                      shape=[None],
                                      name=network_name + '_action'
                                      )
            self.Q_target = tf.placeholder(tf.float32,
                                           shape=[None, num_actions],
                                           name=network_name + '_Q_target'
                                           )
            self.conv1 = tf.layers.conv2d(inputs=self.s_t,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding='valid',
                                          data_format='channels_first',
                                          activation=tf.nn.relu,
                                          name=network_name + '_conv1_layer'
                                          )
            self.conv2 = tf.layers.conv2d(inputs=self.conv1,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding='valid',
                                          activation=tf.nn.relu,
                                          name=network_name + '_conv2_layer'
                                          )
            self.flatten = tf.layers.flatten(self.conv2,
                                             name=network_name + '_flatten'
                                             )
            self.dense = tf.layers.dense(inputs=self.flatten,
                                         units=256,
                                         activation=tf.nn.relu,
                                         name=network_name + '_dense1_layer'
                                         )
            self.Q_values = tf.layers.dense(inputs=self.dense,
                                            units=num_actions,
                                            activation=None,
                                            name=network_name + '_output_layer'
                                            )

            self.best_action = tf.argmax(self.Q_values, 1)
            self.loss = tf.losses.mean_squared_error(self.Q_values,
                                                     self.Q_target)
            self.adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                               name=network_name + '_adam'
                                               )
            self.train = self.adam.minimize(self.loss)

    def update_lr(self):
        """Reduce the learning rate of the Q-Network by 2%.
        """
        self.learning_rate = 0.98*self.learning_rate

        return self.learning_rate

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


def update_graph(variables):
    """Create a list of variable update operations. These operations assign
    weight values from the network named "online" to the one named "target".
    """
    # Get the parameters of our DQNNetwork
    online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "online")

    # Get the parameters of our Target_network
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target")

    update_ops = list()

    # Update our target_network parameters with DQNNetwork parameters
    for online_vars, target_vars in zip(online_vars, target_vars):
        update_ops.append(target_vars.assign(online_vars))
    return update_ops


def update_target(update_ops, session):
    """Update the target network parameters to match those of
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
