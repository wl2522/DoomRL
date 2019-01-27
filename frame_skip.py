"""This is an example script showing the frame skipping algorithm that's used
to add game states to the memory buffer during training. Integers are used to
represent consecutive frames.

The purpose of the algorithm is collect pairs of game states, with each state
consisting of 4 frames.

In particular, the algorithm only stores every 4th frame while ignoring
the rest. Most importantly, there are overlapping frames within each pair of
consecutive states:

    state1 = (1, 4, 8, 16)
    state2 = (4, 8, 16, 20)

    state2 = (4, 8, 16, 20)
    state3 = (8, 16, 20, 24)

We use deques to store these states before they're added to the memory buffer
since they allow elements to be popped from either end of the queue, which
allows us to pop the oldest state from the queue.

"""

from collections import deque

buffer = list()
queue = deque()
experience = deque()

# Push four empty lists into the stack to start the frame-skipping algorithm
for i in range(4):
    queue.append(list())

for frame in range(64):
    # Process only every 4th frame
    if (frame + 1) % 4 == 0:
        for i in range(4):
            queue[i].append(frame//4)

        state1 = queue.popleft()

        # Add states to the buffer once there are 4 frames in the oldest one
        if len(state1) == 4:
            experience.append(state1)

        # Add experiences  to the buffer as pairs of consecutive states

        if len(experience) == 2:
            buffer.append((experience[0], experience[1]))
            # Pop the first state in the queue to make room for the next state
            experience.popleft()

        # Replace the state we just popped with a new one
        queue.append(list())

print(buffer)