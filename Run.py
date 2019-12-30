import gym
import numpy as np
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
import time
env = gym.make("CartPole-v1")

print("Understanding the Environment")
print("\n")
print("Action Space: ",env.action_space)
print("State Space: ",env.observation_space)
print("State Example: ",env.reset())
print("Cycle tuple Example: ",env.step(0))


class My_DQN_Agent():
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=2000)
        
        self.e = 1.000
        self.e_decay = 0.996
        self.e_min = 0.001
        
        self.learning_rate = 0.0005
        self.discount_rate = 0.995
        
        self.model = self.build_model()
        
    def build_model(self):
        
        model = Sequential()
        
        model.add(Dense(20, input_dim=self.state_size, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss = "mse", optimizer = Adam(lr=self.learning_rate))
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state):
        # Pick random Action if rand less than epsilon
        if np.random.rand() <= self.e:
            return np.random.randint(0,self.action_size)
        
        # Otherwise, use the current best options as action
        
        all_q_values = self.model.predict(state)
        return np.argmax(all_q_values[0])
    
    def replay(self, batch_size):
        
        #This will take (batch_size) number of (self.memory) as a mini_batch
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            if done:
                # if it has terminate, reward is just the final reward.
                target = reward
            else:
                # if it has next_state, use the bellman equation to calculate the reward
                
                target = reward + self.discount_rate * np.max(self.model.predict(next_state)[0])
            
            # Now I replay this piece on memory, get the result, then adjust it
            result = self.model.predict(state)
            result[0][action] = target
            
            self.model.fit(state, result, epochs=1, verbose=0)
        
        if self.e > self.e_min:
            self.e *= self.e_decay
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)
        



def test():
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = My_DQN_Agent(state_size,action_size)
    agent.load("cart-pole_v1_weights_0250.hdf5")

    agent.e = 0
    
    done = False
    
    state = env.reset()
    state = state.reshape(1,state_size)
    total_reward = 0
    # This is a continous game, it can go forever, so limit it in 5000 time step.
    for time_steps in range(5000):  
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, _= env.step(action)
        
        reward = reward if not done else -20

        total_reward += reward
        state = next_state
        
        if done:
            # Since we want to see how long can the agent sustain in the game. Time Step is a good measurement.
            print("Episode: {}/{} | epsilon = {:0.2f} | score = {} | Total Reward = {}".format(1, 1, agent.e, time_steps, total_reward))

            break

    


if __name__ == '__main__':
    test()
    
    



    
    