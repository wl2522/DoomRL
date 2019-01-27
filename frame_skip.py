from collections import deque

buffer = list()
queue = deque()
experience = deque()

# Push four empty lists into the stack to start the frame-skipping algorithm
for i in range(4):
    queue.append(list())

# Use a counter to keep track of how many frames have been proccessed
counter = 0

for frame in range(64):
    # Increment the counter first so that we can check for divisibility by 4
    counter += 1
    # Process only every 4th frame
    if counter % 4 == 0:
        for i in range(4):
            queue[i].append(frame//4)


        state1 = queue.popleft()
        # Begin adding states to the buffer once there are 4 frames in the oldest one
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