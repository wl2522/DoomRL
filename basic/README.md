# Basic Scenario

In this scenario, the agent appears in the middle of one side of a room with a pistol while a monster randomly appears somewhere along the wall on the other side of the room. The agent can perform the following actions:

1. move left
2. move right
3. shoot pistol

The agent must shoot the monster once in order to win an episode. The episode automatically ends if the agent fails to do so within 300 time steps. For each time step, the agent receives -1 reward to encourage it to act quickly.

The following table shows the rewards associated with each action in this scenario:

| Action          | Reward |
|-----------------|--------|
| each time step  | -1     |
| shoot and miss  | -6     |
| hit the monster | +100   |

## Network Architecture

1. resize the input array to 80×60×3 and normalize the pixel values
2. 32 filter convolutional layer
3. 64 filter convolutional layer
4. flatten the array
5. 512 unit fully-connected layer
6. mean squared loss
7. Adam optimizer

The agent was trained with a deep Q-Network using Adam as the optimizer. The inputs are the game states (pixels displayed on the screen) captured during each time step. The game states are resized from 640×480 to 80×60 and passed as inputs to two convolutional layers followed by a fully-connected layer. The loss function is the mean-squared error between the Q-values associated with each action at one time step and the Q-values of each action in the subsequent time step after a chosen action has been performed.

## Training

The agent was trained using an ε-greedy strategy according to the following schedule:

| Training Epochs | ε value                   |
|-----------------|---------------------------|
| First 20%       | 1                         |
| 20-90%          | decreases from 0.9 to 0.1 |
| 90-100%         | 0.1                       |

During the first 20% of the training epochs, the agent explores the environment by selecting actions randomly according to a uniform distribution. In the middle phase, ε is initially set to 0.9. The agent greedily chooses an action with 0.1 probability. At each epoch, ε is reduced by a constant amount until it reaches 0.1. In the final phase of training, ε remains at 0.1. The agent greedily chooses an action with 0.9 probability.

## Demo Videos

### agent with no training playing the basic scenario for 20 episodes

[![basic scenario random agent](https://lh3.googleusercontent.com/uJx0vUdRs4BLvRPAKVnmq2RtK3loA_MjPCog4HALs6pqJGKsOMwqNleU6hHW0LHzhZMP2cETfxw=w640)](https://drive.google.com/file/d/1ZqdB9cqy-GbpPF-OY1Cmp6w_hVogZK_p/view)

(Pauses during the video are due to the episode resetting after the agent fails to shoot the monster within 300 time steps.)

### agent trained for 30 epochs playing the basic scenario for 20 episodes

[![basic scenario trained agent](https://lh3.googleusercontent.com/ow9pMBjJ5PlsifZec9U9axNEWWVBnvdgt-0u2GO3VCRI9yMSZBE88xzplINomUlFH-WyEbOVcrA=w640)](https://drive.google.com/file/d/1azoMIdvmOAPBHoQTkVoNA8DRWw9iOCsm/view)
