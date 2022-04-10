import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np

if torch.cuda.is_available():
    device =torch.device('cuda')
else:
    device = torch.device('cpu')

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            #nn.Linear(state_dim, 1024),
            nn.LSTM(1,1024),
            #nn.Linear(1024,1024),
            #nn.ReLU(),
            #nn.Linear(1024,512),
            #nn.ReLU(),
            #nn.Linear(512,256),
            #nn.ReLU(),
            #nn.Linear(256, 16),
            #nn.ReLU(),
            #nn.Linear(16,8),
            nn.Linear(1024, n_actions),
            #nn.Softmax(dim=0)
        )
        self.lstm1 = nn.LSTM(state_dim,128,batch_first=True)
        #self.lstm2 = nn.LSTM(128,32)
        #self.lstm3 = nn.LSTM(32,8)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 16)
        self.linear3 = nn.Linear(16, 8)
        self.linear6 = nn.Linear(8, n_actions)
        self.activation = nn.Softmax(dim=0)
    
    def forward(self, X):
        #X =self.lstm1(X)
        #X =self.lstm2(X)
        out,(hidden,_) =self.lstm1(X)
        out = self.linear1(hidden[-1])
        out=self.linear2(out)
        out=self.linear3(out)
        out=self.linear6(out)
        #out=self.linear(out)
        #out = self.linear2(out)
        #out = self.linear3(out)
        #out = self.linear4(out)
        #out = self.linear5(out)
        #out = self.linear6(out)
        out = self.activation(out)
        #print(out.size())
        return out


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.lstm1 = nn.LSTM(state_dim,128,batch_first=True)
        self.lstm2 = nn.LSTM(128,32)
        self.lstm3 = nn.LSTM(32,8)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64,16)
        self.linear3 = nn.Linear(16, 8)
        self.linear6 = nn.Linear(8, 1)

    
    def forward(self, X):
        #X =self.lstm1(X)
        #X =self.lstm2(X)
        out,(hidden,_) =self.lstm1(X)
        out = self.linear1(hidden[-1])
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear6(out)
        #out = self.linear(out)
        return out



"""Agent class initialized with separate actor and critic networks"""
class Agent:

    def __init__(self, actor_lr, critic_lr, input_dims, gamma=1):
        self.gamma = gamma
        self.actor = Actor(input_dims*2, 1).to(device)
        self.critic = Critic(input_dims*2).to(device)

        # Initializing the learning rates of actor and critic
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.score = None
        self.value = None


    # Computing phi(st) for the value function
    def compute_phi(self, critic_state, probs):
        #print(critic_state.size())
        #print(probs.size())
        x = np.multiply(critic_state.detach().cpu().numpy(), probs.detach().cpu().numpy().reshape(probs.shape[0], 1))
        phi_st = x.sum(axis=0)
        phi_st = phi_st.reshape(1, phi_st.shape[0])
        return phi_st
    
    def timesteps(self,state,action2):
        for i in action2:
            i=torch.Tensor(i)
            i=i.to(device)
            #print(i.size())
            #i=torch.unsqueeze(i,0)
            #i=torch.unsqueeze(i,0)
            i=torch.reshape(i,(1,1,92))
            i=i.repeat(state.size()[0],1,1)
            state = torch.cat((state,i),1)
        return state
    

    # Selecting greedy action while performing testing
    def choose_action_test(self, observation,action2):
        state = torch.from_numpy(observation).float().to(device)

        # Passing the current state to the neural network
        state1 = torch.unsqueeze(state,1)
        state1 = self.timesteps(state1,action2)
        actor_state = self.actor.forward(state1)


        # Getting the probabilities provided by softmax
        probabilities = actor_state
        probabilities = probabilities.reshape(probabilities.shape[0])
        # Choosing the greedy action for testing
        action = torch.argmax(probabilities)
        return action.item()


    # Selection action based on categorical distribution
    def choose_action(self, observation,action2):
        observation = torch.from_numpy(observation).float().to(device)

        # Passing the current state to the neural network
        state = torch.unsqueeze(observation,1)
        state = self.timesteps(state,action2)
        actor_state = self.actor.forward(state)
        
        # Getting the probabilities provided by softmax
        probabilities = actor_state
        probabilities = probabilities.reshape(probabilities.shape[0])

        # Getting the chosen action's score function and storing to use for update later
        action_probs = Categorical(probabilities)
        action = action_probs.sample()
        self.score = action_probs.log_prob(action)

        # Calculating the state for the critic, using probabilities of the actor
        phi_s = self.compute_phi(observation, probabilities)
        phi_s = torch.from_numpy(phi_s).float()
        #print(phi_s.shape)
        phi_s = phi_s.to(device)
        phi_s = torch.unsqueeze(phi_s,1)
        phi_s = self.timesteps(phi_s,action2)
        #print(phi_s.shape)
        # Computing the critic's value function and storing to use for update later
        critic_value = self.critic.forward(phi_s)
        value = critic_value.reshape(critic_value.shape[0])
        self.value = critic_value

        return action.item()


    # Selecting optimal action
    '''
    def choose_best_action(self, observation, rel_label):
        observation = torch.from_numpy(observation).float()
        #observation = torch.unsqueeze(observation,1)
        
        actor_state = self.actor.forward(torch.unsqueeze(observation,1))
        
        probabilities = actor_state
        probabilities = probabilities.reshape(probabilities.shape[0])

        # Getting the chosen action's log probability and storing to update later
        action_probs = Categorical(probabilities)

        action = torch.tensor(np.argmax(rel_label))
        self.score = action_probs.log_prob(action)

        phi_s = self.compute_phi(observation, probabilities)
        phi_s = torch.from_numpy(phi_s).float()

        critic_value = self.critic.forward(phi_s)
        value = critic_value.reshape(critic_value.shape[0])
        self.value = critic_value

        return action.item()

    '''
    # Updating the actor and critic parameters
    def update(self, state, reward, new_state,action2):
        # state for the value function of the current state
        state = torch.from_numpy(state).float().to(device)

        # new_state for the value function of the next state
        new_state = torch.from_numpy(new_state).float().to(device)
        
        # Indicates if trajectory has ended
        done = False
        critic_value_ = 0

        # If not the last state in the trajectory compute the 
        # next state's value function
        if len(new_state) != 0:

            #new_state = torch.unsqueeze(new_state,1)
            #actor_state = self.actor.forward(torch.unsqueeze(new_state,1))

            state1 = torch.unsqueeze(new_state,1)
            #print(state1.size())
            state1 = self.timesteps(state1,action2)
            actor_state = self.actor.forward(state1)
            probabilities = actor_state
            probabilities = probabilities.reshape(probabilities.shape[0])
            phi_s_ = self.compute_phi(new_state, probabilities)
            phi_s_ = torch.from_numpy(phi_s_).float()
            phi_s_ = phi_s_.to(device)
            phi_s_ = torch.unsqueeze(phi_s_,1)
            phi_s_ = self.timesteps(phi_s_,action2)
            critic_value_ = self.critic.forward(phi_s_)
            critic_value_ = critic_value_.reshape(critic_value_.shape[0])

        else:
            done = True

        # Fetching current state's critic value from buffer
        critic_value = self.value
        #critic_value = critic_value.to(device)

        # Converting reward to tensor (to use in-built functions)
        reward = torch.tensor(reward).reshape(1, 1).float().to(device)

        # Using the done flag to omit the value of next state when not present
        advantage = reward + (1-done) * self.gamma * critic_value_ - critic_value

        # Calculate the critic's loss
        critic_loss = advantage.pow(2).mean()

        # Reset the gradients of the critic
        self.adam_critic.zero_grad()

        # Backpropagate the critic loss
        critic_loss.backward()
        self.adam_critic.step()

        # Calculate the actor's loss
        actor_loss = -self.score * advantage.detach()

        # Reset the gradient of the actor
        self.adam_actor.zero_grad()

        # Backpropagate the actor loss
        actor_loss.backward()
        self.adam_actor.step()

        # Reset the buffers
        self.score = None
        self.value = None

