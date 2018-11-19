from collections import deque


class Q_network():
    """Create a Q-network to estimate values and choose actions
    for a given state
    """
    def __init__(self, network_name, height, width, channels, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.s_t = tf.placeholder(tf.float32,
                                  shape=[None, height, width, channels],
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

        self.input_layer = tf.reshape(self.s_t,
                                      [-1, height, width, channels],
                                      name=network_name + '_input_layer'
                                     )
        self.conv1 = tf.layers.conv2d(inputs=self.input_layer,
                                      filters=32,
                                      kernel_size=[8, 8],
                                      strides=[4, 4],
                                      padding='valid',
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
                                        units=len(actions),
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
        self.learning_rate = 0.98*self.learning_rate

        return self.learning_rate

    def calculate_loss(self, session, s, q):
        L, _ = session.run([self.loss, self.train],
                           feed_dict={self.s_t: s,
                                      self.Q_target: q})

        return L

#Return the array of Q-values and the best action associated with a given state

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
    """Create a list of variable update operations.
    """
    update_ops = list()

# Assign weight values from the network created first to the one created second

    for idx, variable in enumerate(variables[:len(variables)//2]):
        op = variable.assign(variables[idx + len(variables)//2].value())
        update_ops.append(op)

    return update_ops


def update_target(update_ops, session):
    """Update the target network parameters to match those of
    the online network.
    """
    for op in update_ops:
        session.run(op)




buffer = list()
stack = deque()
counter = 0

for i in range(7):
    stack.append(list())

for frame in range(16):
    print(frame)
    stack.append(list())
    for i in range(counter, counter + 4):
        stack[i].append(frame)

    counter += 1

    if counter > 6 and counter < 10:
        stack.popleft()

    if len(stack[0]) == 4:
        buffer.append(stack.popleft())
        stack.append(list())


buffer