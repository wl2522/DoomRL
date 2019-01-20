from collections import deque

buffer = list()
stack = deque()
counter = 0

for i in range(4):
    stack.append(list())

for frame in range(64):
    if (counter + 1) % 4 == 0:
        for i in range(counter//4, (counter + 4)//4 + 3):
            stack[i].append(frame)
#        if counter > 6 and counter <= 15:
#            stack.popleft()

        if len(stack[3]) == 4:
            phi = stack.pop()
            buffer.append(phi)
            stack.append(list())
        stack.append(list())

    counter += 1
    print(counter, stack)
print(buffer, stack, counter)
