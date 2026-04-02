import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from constants import ACTIONS, SNAKE_FEATURE_COLS


class QTrainer:
    def __init__(self, model, lr = 0.01, gamma = 0.9):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        #turns all inputs into tensors and adds a dimension for complacency with torch achitecture
        states = torch.tensor(states, dtype=torch.float).reshape(-1, self.model.linear1.in_features)
        next_states = torch.tensor(next_states, dtype=torch.float).reshape(-1, self.model.linear1.in_features)
        actions = torch.tensor(actions, dtype=torch.long).reshape(-1, len(ACTIONS))
        rewards = torch.tensor(rewards, dtype=torch.float).reshape(-1, 1)
        dones = torch.tensor(dones, dtype=torch.bool).reshape(-1, 1)
        #handling for single steps

        #predicted Q values with current state
        pred = self.model(states)
        target = pred.clone()

        with torch.no_grad():
            next_q_values = self.model(next_states)

        for idx in range(len(dones)):
            Q_new = rewards[idx].item()
            if not dones[idx]:
                # Bellman Equation: Q_new = reward + gamma * max(next_predicted_Q_value)
                Q_new = rewards[idx].item() + self.gamma * torch.max(next_q_values[idx]).item()



            action_idx = torch.argmax(actions[idx]).item()
            target[idx][action_idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class ReplayBuffer:
    def __init__(self, max_size = 100):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample (self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)


class Linear_Q(nn.Module):
    def __init__(self, input_size = len(SNAKE_FEATURE_COLS), hidden_size=2**4, output_size=4, solution = None):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.output_size = output_size


        if solution is not None:
            self.solution_to_nn_weights(solution)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
        
    def solution_to_nn_weights(self,solution):
        """unfolds chromosome into lists of neron weights by actions"""
        n_features = len(SNAKE_FEATURE_COLS)

        solution_tensor = torch.tensor(solution, dtype=float)

        hidden_weights_end = n_features*self.hidden_size
        hidden_weights_tensor = solution_tensor[ :hidden_weights_end].view(self.hidden_size,n_features)

        hidden_bias_end = hidden_weights_end + self.hidden_size
        hidden_bias_tensor = solution_tensor [hidden_weights_end : hidden_bias_end]
        
        output_weights_end = hidden_bias_end + (self.hidden_size * self.output_size)
        output_weights_tensor = solution_tensor[hidden_bias_end : output_weights_end].view(self.output_size,self.hidden_size)
        
        output_bias_end = output_weights_end + self.output_size
        output_bias_tensor = solution_tensor[output_weights_end : output_bias_end]
        #changing wieghts without leaving a record
        with torch.no_grad():
            self.linear1.weight.copy_(hidden_weights_tensor)
            self.linear1.bias.copy_(hidden_bias_tensor)
            self.linear2.weight.copy_(output_weights_tensor)
            self.linear2.bias.copy_(output_bias_end)



        

    
    

class Agent:
    def __init__(self,solution = None, decay_actions = 1000, input_size = 10, hidden_size = 10, output_size = 10, epsilon = .5):
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        #smooth decay of epsilon
        self.decay_rate = (self.min_epsilon/self.epsilon)**(1/decay_actions)
        self.learn_rate = 0.01
        self.gamma = 0.9
        self.model = Linear_Q(solution=solution, input_size=input_size,hidden_size=hidden_size, output_size=output_size)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
        self.memory = ReplayBuffer(max_size = 100_000)
        self.batch_size = 100


    def get_action(self, state, use_epsilon = True):
        #for purely evaluating the learned weights
        if not use_epsilon:
            self.epsilon = 0
        # update epsilon based on game count
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        
        #random move
        if random.random() < self.epsilon:
            move = random.randint(0,len(ACTIONS)-1)

        #predicted move
        else:
            state_data = list(state.values()) if isinstance(state, dict) else state
            state_tensor = torch.tensor(state_data, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()

        return move
    def train(self, old_state, action, reward, new_state, done):
            # convert action to a vector then train.
            # this is nessecarry to match q trainer argmax

            old_state_list = list(old_state.values()) if isinstance(old_state, dict) else old_state
            new_state_list = list(new_state.values()) if isinstance(new_state, dict) else new_state

            action_vector = [0]*len(ACTIONS)
            action_vector[action] = 1

            self.memory.push(old_state_list, action_vector, reward, new_state_list, done)
            self.train_memory()
            
            
            self.trainer.train_step([old_state_list], [action_vector], [reward], [new_state_list], [done])


    def train_memory(self):
        if len(self.memory) < self.batch_size:
            return
        

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        self.trainer.train_step(states, actions, rewards, next_states, dones)