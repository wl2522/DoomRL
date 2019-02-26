from collections import deque
import numpy as np


class Buffer:
    """Create a buffer object that holds a set of training experiences
    as state-action-reward tuples.
    """
    def __init__(self, size=1000):
        self.memory = deque(maxlen=size)
        self.length = len(self.memory)

    def add_experience(self, experience):
        """Add a new experience to the buffer.
        Remove the oldest experience in the buffer if it's already full.
        """
        self.memory.append(experience)
        self.length = len(self.memory)

    def sample_buffer(self, sample_size):
        """Return a batch of experience arrays randomly
        sampled from the buffer.

        Parameters
        ----------
        sample_size : int
            the number of experiences to sample from the buffer

        Returns
        -------
        s1 : ndarray
            a batch of game states
        a : ndarray
            a batch of actions that were chosen
        r : ndarray
            a batch of rewards obtained from the chosen actions
        s2 : ndarray
            a batch of game states that follow the ones contained in s1
            (store the first game state if the chosen action ends the episode)
        terminal : ndarray
            an array of integer variables that indicate if the episode is over

        """
        sample = np.random.choice(self.length, size=sample_size, replace=False)
        s1 = np.concatenate([self.memory[idx][0] for idx in sample],
                            axis=0)
        a = np.array([self.memory[idx][1] for idx in sample])
        r = np.array([self.memory[idx][2] for idx in sample])
        s2 = np.concatenate([self.memory[idx][3] for idx in sample],
                            axis=0)
        terminal = np.array([self.memory[idx][4] for idx in sample],
                            dtype=np.int32)

        return s1, a, r, s2, terminal


class FrameQueue:
    """Create a queue that stacks incoming screen buffer frames into
    game states according to the frame skipping algorithm. Each game state
    is then inserted into a separate queue that stores pairs of
    game states.
    Each game state is grouped into an experience tuple
    along with the corresponding action taken at that state,
    the reward received, the next game state, and an indicator variable
    denoting whether the episode has ended or not.
    This experience is then finally added to the memory buffer.
    """
    def __init__(self, stack_len):
        self.stack_len = stack_len
        self.frame_queue = deque([list() for i in range(self.stack_len)],
                                 maxlen=self.stack_len)
        self.experience_queue = deque(maxlen=2)

    def stack_frame(self, frame):
        """Add a new frame into each frame stack in the queue.
        """
        for i in range(self.stack_len):
            self.frame_queue[i].append(frame)

        # Pop and concatenate the oldest stack of frames
        stack = self.frame_queue.popleft()
        stack = np.concatenate(stack, axis=1)

        # Replace the state we just popped with a new one
        self.frame_queue.append(list())

        return stack

    def queue_experience(self, stack, done):
        """Add a game state into the experience prior to it being included
         in an experience tuple. Ignore the first few states of an episode
        which don't contain enough frames to create a game state.

        If a episode has finished and ther aren't enough remaining frames
        to create a full game state, then that terminal game state is padded
        with zero arrays until the sufficient stack length is reached.
        """
        if not done:
            if stack.shape[1] == self.stack_len:
                self.experience_queue.append(stack)
        else:
            if stack.shape[1] < self.stack_len:
                # Infer the height and width of the frame from the input stack
                pad_len = self.stack_len - stack.shape[1]
                pad_shape = (1, pad_len, *stack.shape[-2:])

                stack = np.concatenate((stack, np.zeros(pad_shape)), axis=1)
                self.experience_queue.append(stack)

    def add_to_buffer(self, buffer, action, reward, done):
        """Add the contents of the experience queue to the buffer
        along with the corresponding action, reward, and
        the episode finished indicator variable.
        """
        if len(self.experience_queue) == 2:
            buffer.add_experience((self.experience_queue[0],
                                   action,
                                   reward,
                                   self.experience_queue[1],
                                   done))
