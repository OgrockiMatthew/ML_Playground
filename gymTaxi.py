import gym
from time import sleep
import numpy as np
import random
from IPython.display import clear_output


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        sleep(.1)


def display_learned_behviour():
    #  Choose random state
    example_state = env.reset()
    example_done = False
    frames = []

    while not example_done:
        example_action = np.argmax(q_table[example_state])
        example_state, example_reward, example_done, _ = env.step(example_action)

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': example_state,
            'action': example_action,
            'reward': example_reward
        }
        )

    print_frames(frames)

# Setup
env = gym.make("Taxi-v3").env
epochs = 0
penalties, reward = 0, 0
done = False
q_table = np.zeros([env.observation_space.n, env.action_space.n])  # Build q_table filled with zeros

# Hyperparameters
alpha = 0.1  # learning rate - How much we learn from each test we do.
gamma = 0.6  # discount rate - How much we value future choices vs the current choice
epsilon = 0.1  # Explore rate - How much we explore vs exploit 1 always explore, -1 always exploit

# For plotting metrics
all_epochs = []
all_penalties = []

#  Training
print("Starting to train")
for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        #  explore vs exploit current table.
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        # values from the action taken
        next_state, reward, done, info = env.step(action)

        #  what is currently in our Q-table for that action
        old_value = q_table[state, action]

        #  From the action we just took what is the best choice to do there
        next_max = np.max(q_table[next_state])

        #  learning rate * our choices outcome in q-table
        value_of_action_taken = (1 - alpha) * old_value
        #  learning rate * (reward for action + (discount rate * best options of the next choice))
        value_of_options_provided_by_choice = alpha * (reward + gamma * next_max)
        #  set new q-learning value with info we got
        q_table[state, action] = value_of_action_taken + value_of_options_provided_by_choice

        state = next_state
        epochs += 1

    #  print progress of training
    if i % 100 == 0:
        clear_output(wait=True)
        print("Episode:" + str(i))

print("Training finished.\n")

display_learned_behviour()

