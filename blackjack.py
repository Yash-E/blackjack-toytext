import gym
import random
import numpy as np
from collections import defaultdict

#Epsilon-Greedy Policy Function
def epsilon_greedy_policy(state, q_values, epsilon):
    if random.uniform(0, 1) < epsilon:
        #Explore a random action
        return env.action_space.sample() 
    else:
        #Exploit the best known action
        return np.argmax(q_values[state])

# Q-Value Update Function
def update_q_values(state, action, reward, next_state, q_values, learning_rate, discount_factor):
    #first we find the best possible action for the next state using the current Q-table.
    best_next_action = np.argmax(q_values[next_state])
    #we then look up the Q-value for the next state that would be achieved by taking that best action.
    q_value_next_state = q_values[next_state, best_next_action]
    td_target = reward + discount_factor * q_value_next_state
    td_delta = td_target - q_values[state, action]
    #update the Q-value for the current state and action pair.
    q_values[state, action] += learning_rate * td_delta

# Training Function
def train_agent(episodes, steps, learning_rate, discount_factor, epsilon, decreasing_rate, q_values):
    for episode in range(episodes):
        state = env.reset()[0]
        for step in range(steps):
            action = epsilon_greedy_policy(state, q_values, epsilon)
            next_state, reward, truncated, terminated, _ = env.step(action)
            update_q_values(state, action, reward, next_state, q_values, learning_rate, discount_factor)
            state = next_state
            if truncated or terminated:
                break
        epsilon *= np.exp(-decreasing_rate)
    return q_values


#Evaluation Function
def evaluate_agent(episodes, q_values):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            action = np.argmax(q_values[state]) #always choose best action
            state, reward, truncated, terminated, _ = env.step(action)
            total_rewards += reward
            if truncated or terminated:
                done = True
        print(f"episode {episode}: total  {total_rewards}")
    return total_rewards / episodes  

#Main Function
def main():
    global env
    env = gym.make("Blackjack-v1", render_mode="rgb_array")
    q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    learning_rate = 0.01
    discount_factor = 0.85
    #discount_factor = 0.25
    epsilon = 1
    episodes = 100000
    decreasing_rate = epsilon / (episodes / 2)
    #also tried an exponential decay, did NOT GO WELL
    steps = 10

    trained_q_values = train_agent(episodes, steps, learning_rate, discount_factor, epsilon, decreasing_rate, q_values)

    evaluation_episodes = 1000
    average_reward = evaluate_agent(evaluation_episodes, trained_q_values)
    print(f"Average Reward: {average_reward}")

    env.close()

if __name__ == "__main__":
    main()