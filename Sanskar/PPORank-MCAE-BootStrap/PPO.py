import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from utils import *




# set device to cpu or cuda
device = torch.device('cpu')

################################## PPO Policy ##################################


class RolloutBuffer:
    """Define the buffer class"""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.is_ep_trunc=[]
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.is_ep_trunc[:]


class ActorCritic(nn.Module):
    """Define the actor and critic networks"""
    def __init__(self, state_dim, action_dim, hnodes):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
                        nn.Linear(state_dim, hnodes),
                        nn.Tanh(),
                        # nn.Linear(hnodes, hnodes),
                        # nn.Tanh(),
                        nn.Linear(hnodes, action_dim),
                        nn.Softmax(dim=0)
                    )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, hnodes),
                        nn.Tanh(),
                        # nn.Linear(hnodes, hnodes),
                        # nn.Tanh(),
                        nn.Linear(hnodes, 1)
                    )


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):
        """Given the state returns the action and action probability according to policy"""
        action_probs = self.actor(state)
        action_probs=action_probs.reshape(action_probs.shape[0])
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    
    def compute_phi(self, critic_state, probs):
        """Compute phi(s) by expectation over phi(s,a)"""
        x = np.multiply(critic_state, probs)
        phi_st = x.sum(axis=0)
        phi_st = phi_st.reshape(1, phi_st.shape[0])
        return phi_st

    def evaluate(self, state, action):
        """Given states and actions evaluate the actions and return the action probabilitys,state values,entropys"""
        action_logprobs=[]
        state_values=[]
        dist_entropy=[]
        for s,a in zip(state,action):
            probs = self.actor(s)
            dist = Categorical(probs.reshape(probs.shape[0]))
            action_logprob = dist.log_prob(a)
            entropy = dist.entropy()
            phi_s=self.compute_phi(s.detach(),probs.detach())
            state_value = self.critic(phi_s)
            action_logprobs.append(action_logprob)
            state_values.append(state_value)
            dist_entropy.append(entropy)
        
        action_logprobs=torch.stack(action_logprobs)
        state_values=torch.stack(state_values)
        dist_entropy=torch.stack(dist_entropy)
        
        return action_logprobs, state_values, dist_entropy

    def evaluate_state(self, s):
        """Given states returns the state values"""
        probs = self.actor(s)
        phi_s=self.compute_phi(s.detach(),probs.detach())
        state_value = self.critic(phi_s)
        
        return state_value


class PPO:
    def __init__(self, state_dim, action_dim,hnodes, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        """Initialize all required objects"""
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim,hnodes).to(device)
        self.optimizer = torch.optim.SGD([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim,hnodes).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def select_action(self, state):
        """Given state select action according to the policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
   
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def select_action_test(self, state):
        """Given state select action according to policy greedily"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action_probs = self.policy.actor(state)
            action_probs=action_probs.reshape(action_probs.shape[0])
            dist = Categorical(action_probs)

            action = np.argmax(action_probs)


        return action


    def update(self):
        """Update the policy and value network by optimizing the objective function"""

        # Monte Carlo estimate of returns
        rewards = []
        
        #bootstrap from final state in the buffer according to whether it is a terminal or not
        if self.buffer.is_terminals[-1]:
           discounted_reward = 0
        else:
           discounted_reward = self.policy_old.evaluate_state(self.buffer.states[-1]).item()
        
        #Run the buffers in reverse and bootstrap whenever an episode ends
        for reward,state,is_terminal,is_ep_trunc in zip(reversed(self.buffer.rewards),reversed(self.buffer.states),reversed(self.buffer.is_terminals),reversed(self.buffer.is_ep_trunc)):
            if is_terminal:
                discounted_reward = 0
            elif is_ep_trunc:
                discounted_reward= self.policy_old.evaluate_state(state).item()

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        # convert list to tensor
        old_states = self.buffer.states
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
    
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
    
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            # print('loss is:' , (loss.mean()))
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
  
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        """Save the current policy in the given path"""
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        """load the policy from the given path"""
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        