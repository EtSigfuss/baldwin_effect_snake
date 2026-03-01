import random

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

    def train_step(self, state, action, reward, next_state, done):
        #turns all inputs into tensors and adds a dimension for complacency with torch achitecture
        state = torch.tensor(list(state.values()), dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(list(next_state.values()), dtype=torch.float).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        done = [done]
        #handling for single steps

        #predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Bellman Equation: Q_new = reward + gamma * max(next_predicted_Q_value)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class Linear_Q(nn.Module):
    def __init__(self, input_size = len(SNAKE_FEATURE_COLS), hidden_size=2^4, output_size=4, solution = None):
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
    def __init__(self,solution = None, decay_actions = 100, input_size = 10, hidden_size = 10, output_size = 10, epsilon = .5):
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        #smooth decay of epsilon
        self.decay_rate = (self.min_epsilon/self.epsilon)**(1/decay_actions)
        self.learn_rate = 0.01
        self.gamma = 0.9
        self.model = Linear_Q(solution=solution, input_size=input_size,hidden_size=hidden_size, output_size=output_size)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)


    def get_action(self, state, use_epsilon = True):
        #for purely evaluating the learned weights
        if not use_epsilon:
            self.epsilon = 0
        # Update epsilon based on game count
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        
        #random move
        if random.random() < self.epsilon:
            move = random.randint(0,len(ACTIONS)-1)

        #predicted move
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()

        return move
    def train(self, old_state, action, reward, new_state, done):
            # convert action to a vector then train.
            # this is nessecarry to match q trainer argmax
            action_vector = [0]*len(ACTIONS)
            action_vector[action] = 1
            
            self.trainer.train_step(old_state, action_vector, reward, new_state, done)