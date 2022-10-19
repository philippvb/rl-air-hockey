import os
from radam import RAdam

# The agent class
from feedforward import FeedForwardNetwork
import torch
import itertools
import numpy as np

class TD3Agent():

    def __init__(self, critic_layers, actor_layers, optimizer="Adam", weight_decay=0, policy_noise=0.1, policy_noise_clip=0.5, gamma=0.9, delay=2, tau=0.005, lr=3e-4, max_action=1):
        # two critics with targets
        self.Q_1 = FeedForwardNetwork(critic_layers, actor=False)
        self.Q_2 = FeedForwardNetwork(critic_layers, actor=False)
        self.Q_1_target = FeedForwardNetwork(critic_layers, actor=False)
        self.Q_2_target = FeedForwardNetwork(critic_layers, actor=False)
        # actor with target
        self.policy = FeedForwardNetwork(actor_layers)
        self.policy_target = FeedForwardNetwork(actor_layers)
        self.policy_noise=policy_noise
        self.policy_noise_clip=policy_noise_clip

        # optimizer
        if optimizer == "SGD_momentum":
            self.Q_optimizer=torch.optim.SGD(list(self.Q_1.parameters()) + list(self.Q_2.parameters()), lr=lr, momentum=0.9)
            self.policy_optimizer=torch.optim.SGD(self.policy.parameters(), lr=lr, momentum=0.9)

        elif optimizer == "RAdam":
            self.Q_optimizer=RAdam(list(self.Q_1.parameters()) + list(self.Q_2.parameters()), lr=lr, weight_decay=weight_decay)
            self.policy_optimizer= RAdam(self.policy.parameters(), lr=lr, weight_decay=weight_decay)
    
        elif optimizer == "Adam":
            self.Q_optimizer=torch.optim.Adam(list(self.Q_1.parameters()) + list(self.Q_2.parameters()), lr=lr, weight_decay=weight_decay)
            self.policy_optimizer= torch.optim.Adam(self.policy.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("Optimizer not found")

        # other hyperparameters
        self.tau=tau
        self.gamma = gamma
        self.delay=delay
        self.train_iteration=0
        self.loss=torch.nn.MSELoss()
        self.max_action=max_action


    def act(self, state):
        """Returns the action for a given state

        Args:
            state (torch.tensor): the states to return the action for, given rowwise

        Returns:
            [torch.tensor]: actions
        """
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return self.policy(state).detach().numpy()


    def train(self, states, actions, next_states, reward, done):
        """Trains the agent on the given batch

        Args:
            states (torch.tensor): 
            actions (torch.tensor): 
            next_states (torch.tensor): 
            reward (torch.tensor): 
            done (torch.tensor): Done flag, 1 indicates done 

        Returns:
            [float]: The mean critic loss for the batch
        """

        # choose next action with noise and compute Td-Target for Q-Update

        with torch.no_grad():
            next_actions = (self.policy_target(next_states) + (torch.rand_like(actions)*self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)).clamp(-self.max_action, self.max_action)
            y = reward + self.gamma * torch.min(self.Q_1_target(torch.cat((next_states,next_actions), dim=1)), self.Q_2_target(torch.cat((next_states,next_actions), dim=1))) * (1-done)

        # compute loss for critic
        critic_loss = self.loss(self.Q_1(torch.cat((states,actions), dim=1)), y) + self.loss(self.Q_2(torch.cat((states,actions), dim=1)), y)
        self.Q_optimizer.zero_grad()
        critic_loss.backward()
        self.Q_optimizer.step()

        # update policy
        if self.train_iteration % self.delay == 0:
            
            # compute loss for actor
            actor_loss = - self.Q_1(torch.cat((states,self.policy(states)), dim=1)).mean()
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()
            # update targets
            self.update_target()

        self.train_iteration+=1

        return critic_loss/2



    def update_target(self):
        """Updates the targets of the actor and critic
        """
        for target_param, param in zip(self.Q_1_target.parameters(), self.Q_1.parameters()):
            target_param.data.copy_(self.tau*param.data + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.Q_2_target.parameters(), self.Q_2.parameters()):
            target_param.data.copy_(self.tau*param.data + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau*param.data + target_param.data*(1.0 - self.tau))

    def save(self, filepath):
        """Saves the states of the networks into the given directory

        Args:
            filepath (string): path to the directory
        """
        if not os.path.exists(filepath):
            os.mkdir(filepath)
            
        torch.save(self.Q_1.state_dict(), filepath + "/Q_1_network")
        torch.save(self.Q_1_target.state_dict(), filepath + "/Q_1_target_network")
        torch.save(self.Q_2.state_dict(), filepath + "/Q_2_network")
        torch.save(self.Q_2_target.state_dict(), filepath + "/Q_2_target_network")
        torch.save(self.policy.state_dict(), filepath + "/Policy_network")
        torch.save(self.policy_target.state_dict(), filepath + "/Policy_target_network")

    def load(self, filepath):
        """Loads the states of the networks from the given directory.

        Args:
            filepath (string): path to the directory
        """
        self.Q_1.load_state_dict(torch.load(filepath + "/Q_1_network"))
        self.Q_2.load_state_dict(torch.load(filepath + "/Q_2_network"))
        self.Q_1_target.load_state_dict(torch.load(filepath + "/Q_1_target_network"))
        self.Q_2_target.load_state_dict(torch.load(filepath + "/Q_2_target_network"))
        self.policy.load_state_dict(torch.load(filepath + "/Policy_network"))
        self.policy_target.load_state_dict(torch.load(filepath + "/Policy_target_network"))




