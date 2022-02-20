import itertools
import gym
import numpy as np
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import namedlist
from itertools import count

# global settings
seed = 114514
render = True

def auto2tensor(fn):
    '''
    decorator that automatically do Tensor()\n
    appliable only for member functions
    '''
    def wrapped(self, x):
        if not isinstance(x, torch.Tensor):
            x = Tensor(x)
        return fn(self, x)
    return wrapped

running_stats = namedlist.namedlist("running_stats", ["n", "mean", "var"])

Transition = namedlist.namedlist('Transition', ['state', 'action',  'a_prob', 'reward', 'next_state', 'mask'])
# mask = 0 if it's a terminal else 1, so rewards from different episodes won't be mixed

class Actor(nn.Module):
    def __init__(self, num_state, hidden_size, num_action):
        super(Actor, self).__init__()
        
        self.fc_stack = nn.Sequential(
            nn.Linear(num_state, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_action)
        )

        # orth init (not sure)
        for layer in self.fc_stack:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

    @auto2tensor
    def forward(self, x):
        x = self.fc_stack(x)
        x = F.softmax(x, dim = 1)
        return x

    def select_action(self, state):
        '''
        only for single dim-1 state
        '''
        with torch.no_grad():
            action_problist = self(state.reshape(1,-1)).reshape(-1)
        action = Categorical(action_problist).sample()
        return action.item(), action_problist[action.item()].item()

class Critic(nn.Module):
    def __init__(self, num_state, hidden_size):
        super(Critic, self).__init__()
        
        self.fc_stack = nn.Sequential(
            nn.Linear(num_state, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # orth init (not sure)
        for layer in self.fc_stack:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

    @auto2tensor
    def forward(self, x):
        x = self.fc_stack(x)
        return x

    def get_value(self, state):
        with torch.no_grad():
            return self(state)

class Memory:
    def __init__(self, num_state): # num_state actually means shape of state
        self.clear()
        self.obs_stat = running_stats(0, np.zeros(num_state), np.zeros(num_state))

    def __len__(self):
        return len(self.mem)

    def append(self, trans):
        self.mem.append(trans)
        self.cnt += 1

        new_obs = np.asarray(trans.state)
        self.obs_stat.n += 1
        if self.obs_stat.n == 1:
            self.obs_stat.mean = new_obs
        else:
            oldmean = self.obs_stat.mean.copy()
            self.obs_stat.mean = self.obs_stat.mean + (new_obs - self.obs_stat.mean) / self.obs_stat.n
            self.obs_stat.var = ((self.obs_stat.n - 2) * self.obs_stat.var + (new_obs - oldmean) * (new_obs - self.obs_stat.mean)) / (self.obs_stat.n - 1)
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            # https://math.stackexchange.com/questions/2798082/how-is-welfords-algorithm-derived


    def sample(self):
        '''
        return Tensors of each elem of Transition\n
        note that states & rewards are normalized
        '''
        fullsample = Transition(*zip(*self.mem))
        fullsample = [np.stack(elem) for elem in fullsample]

        return Transition(*fullsample)

    def clear(self):
        self.mem = []
        self.cnt = 0

def normclip(x, clip = float("inf")):
    return  torch.clamp((x - torch.mean(x)) / max(torch.std(x), 1e-7), -clip, clip)

def savenet(net, filename):
    torch.save(net.state_dict(), filename)

class PPO():
    gamma = 0.99
    lamda = 0.97
    
    clip_param = 0.2

    T_timesteps = 1024
    K_epochs = 3
    M_minibatchsize = 256

    hidden_size = 64 # not sure

    def __init__(self, env_name):
        self.env = gym.make(env_name)

        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.n # might be different in different env

        self.memory = Memory(num_state)

        self.env.seed(seed)
        torch.manual_seed(seed)

        self.actor = Actor(num_state, self.hidden_size, num_action)
        self.critic = Critic(num_state,self.hidden_size)

        self.optim = optim.Adam(itertools.chain(self.actor.parameters(), self.critic.parameters()), lr = 3e-4)

    def traj_gen(self, i_ep):
        '''
        generate some trajectories\n
        stored in self.memory\n
        return the index of episode
        '''
        self.memory.clear()
        for i in count(i_ep):
            state = self.env.reset()
            if render : self.env.render()

            for _ in count():
                state = (state - self.memory.obs_stat.mean) / np.fmax(np.sqrt(self.memory.obs_stat.var), 1e-7)
                state = np.clip(state, -5.0, 5.0)

                action, action_prob = self.actor.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if render : self.env.render()

                # states stored are all normalized & clipped
                trans = Transition(state, action, action_prob, reward,  next_state, 0 if done else 1)

                self.memory.append(trans)
                state = next_state

                if len(self.memory) >= self.T_timesteps:
                    print(f"traj generation done...\n"
                        + f"episode index : {i_ep} --> {i}\n")
                    return i

                if done:
                    break # reset env, goto next episode
        

    def update(self, i_epoch):
        '''
        train (assume trajectories have been generated)\n
        i_epoch = the num of trained epochs\n
        '''
        # step 1 : get records in memory

        stat = self.memory.sample() # normalized, len == batchsize (according to the paper)

        # step 3 : get approximation of advantages (GAE) from records,then normalize the advantages
        
        G = Tensor(self.T_timesteps)
        deltas = Tensor(self.T_timesteps)
        A = Tensor(self.T_timesteps)
        values = self.critic.get_value(stat.state).reshape(-1)
        # the end of stat may not be terminal, so prev_stuff need to be estimated
        # assume state[self.T_timesteps - 1] approx= state[self.T_timesteps]
        prev_g = values[self.T_timesteps - 1]
        prev_v = values[self.T_timesteps - 1]
        prev_A = 0
        
        for t in reversed(range(self.T_timesteps)):
            G[t] = stat.reward[t] + self.gamma * prev_g * stat.mask[t]
            deltas[t] = stat.reward[t] + self.gamma * prev_v * stat.mask[t] - values[t]
            A[t] = deltas[t] + self.gamma * self.lamda * prev_A * stat.mask[t]

            prev_g = G[t]
            prev_v = values[t]
            prev_A = A[t]

        A = normclip(A) # should A be normalized?

        # step 4 : optimize LOSS with minibatchsize = M, each sample point should be used in K_epoches
        for _ in range(self.K_epochs * int(self.T_timesteps / self.M_minibatchsize)):

            # for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), self.M_minibatchsize, False):
            for i in range(0, self.T_timesteps, self.M_minibatchsize):

                index = range(i, i + self.M_minibatchsize)

                # detach?
                advantages = A[index]
                # advantages.detach_()

                # get necessary data
                value_from_statistics = G[index]
                value_from_critic = self.critic(stat.state[index]).reshape(-1)

                old_a_prob = Tensor(stat.a_prob[index])
                new_a_prob = Tensor(self.actor(stat.state[index])).gather(1, Tensor(stat.action[index]).to(torch.int64).reshape(-1,1)).reshape(-1)

                ratio = new_a_prob / old_a_prob

                # calculate LOSSES (with original notation from the paper)
                L_CPI = ratio * advantages
                L_CPI_clip = ratio.clamp(1-self.clip_param, 1+self.clip_param) * advantages
                L_CLIP = torch.min(L_CPI, L_CPI_clip).mean()

                L_VF = (value_from_critic - value_from_statistics).pow(2).mean()

                S = -(new_a_prob * torch.log(new_a_prob)).mean()

                loss = -(L_CLIP - 0.5 * L_VF + 0.01 * S)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # show info on terminal
                i_epoch += 1
                if i_epoch % 100 == 0:
                    print(f"{i_epoch} epochs trained...\n"
                    +     f"  mean G in recent T timesteps : {G.mean():.4f}\n"
                    +     f"  total loss\t||\tL_CLIP\t|\tL_VF\t|\tS\t\n"
                    +     f"  {loss:.4f}\t||\t{L_CLIP:.4f}\t|\t{L_VF:.2f}\t|\t{S:.4f}\t\n")                
        return i_epoch


def demo():
    i_episode = 0
    i_epoch = 0
    agent = PPO("CartPole-v0")
    while i_epoch < 1000:
        i_episode = agent.traj_gen(i_episode)
        i_epoch = agent.update(i_epoch)

if __name__ == '__main__':
    demo()
