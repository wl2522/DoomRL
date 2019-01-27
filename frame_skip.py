from collections import deque

buffer = list()
queue = deque()

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


        phi = queue.popleft()
        # Begin adding stacks to the buffer once there are 4 in the oldest one
        if len(phi) == 4:
            buffer.append(phi)
        # Replace the stack we just popped with a new one
        queue.append(list())

print(buffer)
