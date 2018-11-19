import numpy as np

class Buffer():
    """Create a buffer object that holds a set of training experiences
    as state-action-reward tuples.
    """
    def __init__(self, size=1000):
        self.buffer = list()
        self.length = len(self.buffer)
        self.size = size

    def add_experience(self, experience):
        """Add a new experience to the buffer.
        Remove the oldest experience in the buffer if it's already full.
        """
        if self.length + 1 >= self.size:
            self.buffer[0:(self.length + 1) - self.size] = []

        self.buffer.append(experience)
        self.length = len(self.buffer)

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
        sample = np.random.randint(self.length, size=sample_size)
        s1 = np.concatenate([self.buffer[idx][0] for idx in sample],
                            axis=0)
        a = np.array([self.buffer[idx][1] for idx in sample])
        r = np.array([self.buffer[idx][2] for idx in sample])
        s2 = np.concatenate([self.buffer[idx][3] for idx in sample],
                            axis=0)
        terminal = np.array([self.buffer[idx][4] for idx in sample],
                            dtype=np.int32)

        return s1, a, r, s2, terminal
