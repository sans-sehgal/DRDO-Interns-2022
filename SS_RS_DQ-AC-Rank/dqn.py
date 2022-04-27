import torch
import torch as T
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from scipy.stats import rankdata
class DeepQNetwork(nn.Module):
    """Deep Q Network"""
    def __init__(self, lr_critic, input_dims,n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        #Define the network
        self.net= nn.Sequential(nn.Linear(*self.input_dims, self.n_actions))
        
        #Define the optimizer
        self.optimizer_critic = optim.Adam(self.parameters(), lr=lr_critic)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)
        nn.init.constant_(self.net[0].weight,1)

    def forward(self, state):
        actions=self.net(state)
        return actions

class ActorNetwork(nn.Module):
    """Actor network"""
    def __init__(self, lr_actor,input_dims,n_actions):
        super(ActorNetwork, self).__init__()

        #Define the network
        self.actor = nn.Sequential(nn.Linear(*input_dims, n_actions),nn.Softmax(dim=0))
        
        #Define the optimizer
        self.optimizer_actor = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = T.device('cpu')
        self.to(self.device)
        nn.init.constant_(self.actor[0].weight,1)

    def forward(self, state):
        actions=self.actor(state)
        return actions

class Agent():
    """Define the RL Agent"""
    def __init__(self, gamma, epsilon, lr_critic,lr_actor, input_dims, batch_size, n_actions,
            max_mem_size=100000,hnodes=256, eps_end=0.05, eps_dec=5e-4,replace_target=100):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = replace_target
        self.hnodes=hnodes

        #self.lr_critic = 0.01

        #Initialize the Evaluation and target networks
        self.Q_eval = DeepQNetwork(lr_critic, n_actions=n_actions, input_dims=input_dims)
        self.Q_next = DeepQNetwork(lr_critic, n_actions=n_actions, input_dims=input_dims)

        #Initialize the Actor Network
        self.Actor = ActorNetwork(lr_actor=lr_actor, input_dims=input_dims, n_actions=n_actions) 

        #Initialize the buffers
        self.state_memory = [0]*self.mem_size
        self.new_state_memory = [0]*self.mem_size
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        """Store the current transition in the buffer"""
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action_dqn(self, observation):
        """Perform episilon greedy selection of action to take(policy in dqn)."""
        '''
        if np.random.random() > self.epsilon:
            state = observation.float().to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(len(observation))

        return action
        '''
        state = observation.float().to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
            #action=actions
        if np.random.random() > self.epsilon:  
            action=rankdata(actions.detach().numpy(),method='ordinal')

        else:
            print('random')
            action = random.sample(range(1,len(actions)+1),len(actions))

        return action

    def choose_action_actor(self, observation):
        """Picks which action to take based on actor policy"""
        with torch.no_grad():
            action_probs = self.Actor(observation)
            action_probs=action_probs.reshape(action_probs.shape[0])
            dist = Categorical(action_probs)

            action = dist.sample()
        
        return action.detach()

    def actor_evaluate(self,state_batch,action_batch):
        """Evaluate the actions taken in the current mini-batch and return the log-probs"""
        action_logprobs=[]
        for s,a in zip(state_batch,action_batch):
            probs = self.Actor(s)
            dist = Categorical(probs.reshape(probs.shape[0]))
            action_logprob = dist.log_prob(torch.Tensor([a]))
            action_logprobs.append(action_logprob)
            #print(action_logprobs)
            #print('------')
        action_logprobs=torch.stack(action_logprobs)
        
        return action_logprobs


    def choose_action_test(self, observation):
        """Pick greedy action wrt to Q network for testing"""
        with torch.no_grad():
            state = observation.float().to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        
        return action

    def choose_action_test_actor(self, state):
        """Pick greedy action wrt to actor policy for testing"""
        with torch.no_grad():
            state = state.float().to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action=rankdata(actions.detach().numpy(),method='ordinal')

        return action
    
    def qvals_minibatch(self,state_batch,action_batch):
        """Returns the qvals of the actions taken in that batch"""
        qvals=[]
        for s,a in zip(state_batch,action_batch):
            qvalues = self.Q_eval.forward(s)
            qvals.append(qvalues[a])

        return T.squeeze(T.stack(qvals))

    def qvals_minibatch_target(self,state_batch):
        """Returns the qvals for every action in the given states"""
        qvals=[]
        for s in state_batch:
            if(len(s)>0):
                qvalues = self.Q_next.forward(s)
                a = T.argmax(qvalues).item()
                qvals.append(qvalues[a])
            else:
                qvals.append(T.tensor([0.0],requires_grad=True))
 
        return T.squeeze(T.stack(qvals))

    

    def learn(self):
        """Samples a minibatch and updates the q-networks"""

        #If the transitions in the buffer less than batch size then return
        if self.mem_cntr < self.batch_size:
            return
        
        #Find no. of transitions in the buffer
        max_mem = min(self.mem_cntr, self.mem_size)

        #Pick a random mini batch
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        #print("Mini Batch:",batch)

        #Get the current mini-batch
        state_batch=[]
        new_state_batch=[]
        for index in batch:
            state_batch.append(self.state_memory[index])
            new_state_batch.append(self.new_state_memory[index])
        
        state_batch=state_batch
        new_state_batch = new_state_batch
        action_batch = self.action_memory[batch]
        
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        #Get the log probabilitys of the current mini-batch
        log_probs=self.actor_evaluate(state_batch,action_batch)

        #Get the qvals 
        q_eval=self.qvals_minibatch(state_batch,action_batch)
        q_next = self.qvals_minibatch_target(new_state_batch)
        q_next[terminal_batch] = 0.0

        #Compute the q-learning target
        q_target = reward_batch + self.gamma*q_next

        #Compute the critic loss
        loss_critic = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        #ret_loss=loss_critic.detach().item()
        #print('------BEFORE------')
        #for name,param in self.Q_eval.named_parameters():
        #    print(name,param.data)
        #Clear any gradients
        self.Q_eval.optimizer_critic.zero_grad()
        self.Actor.optimizer_actor.zero_grad()

        #Update critic parameters by SGD and backprop
        loss_critic.backward()
        self.Q_eval.optimizer_critic.step()
        #print('-----LOSSS----------')
        #print(loss_critic)
        #Compute the actor loss
        loss_actor = (-log_probs*self.qvals_minibatch(state_batch,action_batch).detach()).mean().to(self.Actor.device)
        #print('------AFTER------')
        #for name,param in self.Q_eval.named_parameters():
        #    print(name,param.data)
        #Update actor parameters by SGD and backprop
        loss_actor.backward()
        self.Actor.optimizer_actor.step()

        #Increase the update counter
        self.iter_cntr += 1

        #Decay epsilon 
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                       else self.eps_min

        #Update the target network
        if self.iter_cntr % self.replace_target == 0:
          self.Q_next.load_state_dict(self.Q_eval.state_dict())

        #return ret_loss
