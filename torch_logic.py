import random

import torch
import torch.nn as nn
import torch.optim as optim

from constants import SNAKE_FEATURE_COLS, episodes_per_life


class QTrainer:
    def __init__(self, model, lr = 0.01, gamma = 0.9):
        self.lr = lr
        self.gamma = gamma # Discount rate (usually 0.9)
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        #turns all inputs into tensors and adds a dimension for complacency with torch achitecturte
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
    def __init__(self, input_size = len(SNAKE_FEATURE_COLS), hidden_size=2**8, output_size=4, solution = None):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        if solution is not None:
            self.load_solution(solution)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def load_solution(self, solution):
        #convert genome into nn???
        pass
        


    
    

class Agent:
    def __init__(self,model_weights = None):
        self.epsilon = 1.0  # Start with 100% exploration
        self.min_epsilon = 0.001
        #smooth decay of epsilon
        self.decay_rate = (self.min_epsilon/self.epsilon)**(1/episodes_per_life)
        self.learn_rate = 0.01
        self.gamma = 0.9
        self.model = Linear_Q()
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)


    def get_action(self, state):
        # Update epsilon based on game count
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        
        # 1. Random Move 
        if random.random() < self.epsilon:
            move = random.randint(0, 3)
        
        # 2. Predicted Move
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()

        return move
    def train(self, old_state, action, reward, new_state, done):
            # Convert action to a vector then train.
            # This is nessecarry to match q trainer argmax
            action_vector = [0, 0, 0, 0]
            action_vector[action] = 1
            
            self.trainer.train_step(old_state, action_vector, reward, new_state, done)