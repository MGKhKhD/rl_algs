import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.autograd as autograd

import numpy as np
import math
import os
import scipy.optimize
import random
from collections import namedtuple
from itertools import count
import argparse

import gym

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'reward'))

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

class Memory:
    def __init__(self):
        self.storage = []
        
    def push(self, *args):
        self.storage.append(Transition(*args))
        
    def sample(self):
        return Transition(*zip(*self.storage))
    
    def __len__(self):
        return len(self.storage)
    
class Running_Stat:
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
        
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - self._M) * (x - oldM)
            
    @property
    def n(self):
        return self._n
    
    @property
    def mean(self):
        return self._M
    
    @property
    def var(self):
        return self._S / (self._n -1) if self._n > 1 else np.square(self._M)
    
    @property
    def std(self):
        return np.sqrt(self.var)
    
    @property
    def shape(self):
        return self._M.shape
    
class Z_Filter:
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        
        self.rs = Running_Stat(shape)
        
    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
            
        if self.demean:
            x = x - self.rs.mean
            
        if self.destd:
            x = x / (self.rs.std + 1e-10)
            
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
    def output_shape(self, input_space):
        return input_space.shape


def mlp(sizes, activation, out_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else out_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class Gaussian_Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_sizes=(400, 300), activation=nn.Tanh):
        super(Gaussian_Policy, self).__init__()
        self.mean = mlp([num_inputs] + list(hidden_sizes) + [num_outputs], activation)
        self.log_std = nn.Parameter(torch.zeros(1, num_outputs))
        
    def forward(self, state):
        mean, log_std, std = self._get_mean_std(state)
        return torch.normal(mean, std)

    def log_density(self, states, actions, volatile=False):
        if volatile:
            with torch.no_grad():
                mean, log_std, std = self._get_mean_std(states)
        else:
            mean, log_std, std = self._get_mean_std(states)
        var = std.pow(2)
        _log_density = -(actions - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
        return _log_density.sum(1, keepdim=True)

    def kl_div_current_new(self, states):
        mean, log_std, std = self._get_mean_std(states)
        mean_new = Variable(mean.data)
        log_std_new = Variable(log_std.data)
        std_new = Variable(std.data)

        kl = log_std - log_std_new + (std_new.pow(2) + (mean_new - mean).pow(2)) / (2.0 * std.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
        
    def _get_mean_std(self, state):
        mean = self.mean(state)
        log_std = self.log_std.expand_as(mean)    
        
        return mean, log_std, torch.exp(log_std)

    def get_flat_params(self):
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params)

    def set_flat_params(self, flat_params):
        idx = 0
        for p in self.parameters():
            l = int(np.prod(list(p.size())))
            p.data.copy_(flat_params[idx: idx + l].view(p.size()))
            idx += l
            
    def get_flat_grad(self, grad_grad=False):
        grads = []
        for p in self.parameters():
            if grad_grad:
                grads.append(p.grad.view(-1))
            else:
                grads.append(p.gard.view(-1))
        return torch.cat(grads)
    
class Value(nn.Module):
    def __init__(self, num_inputs, hidden_sizes=(400, 300), activation=nn.Tanh):
        super(Value, self).__init__()
        self.value = mlp([num_inputs] + list(hidden_sizes) + [1], activation)
        
    def forward(self, state):
        return self.value(state)

    def regularize_loss(self, loss, l2_reg=0.001):
        ll = 0
        for p in self.parameters():
            ll += p.data.pow(2).sum()
        return loss + ll * l2_reg
    
class TRPO(nn.Module):
    def __init__(self, num_inputs, num_outputs, value_lr=0.001, discount=0.99, tau=0.97, max_kl=0.03, l2_reg=0.001, damping=0.1):
        super(TRPO, self).__init__()
        self.policy_net = Gaussian_Policy(num_inputs, num_outputs)
        self.value_net = Value(num_inputs)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)

        self.discount = discount
        self.tau = tau
        self.max_kl = max_kl
        self.l2_reg = l2_reg
        self.damping = damping

        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.policy_net(state)

    def _extract_info(self, batch):
        rewards = torch.FloatTensor(batch.reward)
        masks = torch.FloatTensor(batch.mask)
        actions = torch.FloatTensor(np.concatenate(batch.action, 0))
        states = torch.FloatTensor(batch.state)
        values = self.value_net(states)
        
        size = actions.size(0)
        returns = torch.FloatTensor(size, 1)
        advantages = torch.FloatTensor(size, 1)
        
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        
        for i in reversed(range(size)):
            returns[i] = rewards[i] + self.discount * prev_return * masks[i]
            deltas = rewards[i] + self.discount * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas + self.discount * prev_advantage * self.tau * masks[i]
            
            prev_return = returns[i, 0]
            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]

        value_loss = (values - returns).pow(2).mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return states, actions, value_loss, advantages
    
    def train(self, batch):
        states, actions, value_loss, advantages = self._extract_info(batch)

        self.value_optimizer.zero_grad()
        value_loss = self.value_net.regularize_loss(loss=value_loss, l2_reg=l2_reg)   
        value_loss.backward()
        self.value_optimizer.step()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        current_log_prob = self.policy_net.log_density(states, actions, volatile=False).data.clone()
        
        def get_loss(volatile=False):
            log_prob = self.policy_net.log_density(states, actions, volatile=volatile)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(current_log_prob))
            return action_loss.mean()
        
        def get_kl():
            return self.policy_net.kl_div_current_new(states)
        
        self._trpo_step(get_loss, get_kl)

    def _trpo_step(self, get_loss, get_kl):

        def conjugate_grad(Avp, b, n_step, residual_tol=1e-10):
            x = torch.zeros(b.size())
            r = b.clone()
            p = b.clone()
            
            rdotr = torch.dot(r, r)
            for i in range(n_step):
                _Avp = Avp(p)
                alpha = rdotr / torch.dot(p, _Avp)
                x += alpha * p
                r -= alpha * _Avp
                new_rdotr = torch.dot(r, r)
                beta = new_rdotr / rdotr
                p = r + beta * p
                rdotr = new_rdotr
                if rdotr < residual_tol:
                    break
            return x

        def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
            fval = f(True).data
            print("fval before line search: ", fval.item())
            for (_n, stepfrac) in enumerate(0.5**(np.arange(max_backtracks))):
                x_new = x + stepfrac * fullstep
                self.policy_net.set_flat_params(x_new)
                new_fval = f(True).data
                actual_improve = fval - new_fval
                expected_improve = expected_improve_rate * stepfrac
                ratio = actual_improve / expected_improve
                print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())
                
                if ratio.item() > accept_ratio and actual_improve.item() > 0:
                    print("fval after line search: ", new_fval.item())
                    return True, x_new
            return False, x
 
        def Fvp(v):
            kl = get_kl().mean()
            
            grads = autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
            
            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = autograd.grad(kl_v, self.policy_net.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
            
            return flat_grad_grad_kl + v * self.damping
        
        loss = get_loss()
        grads = autograd.grad(loss, self.policy_net.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        stepdir = conjugate_grad(Fvp, -loss_grad, 10)
        
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm[0]
        
        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        print("Lagrange multiplier ", lm[0], 'grad_norm: ', loss_grad.norm())
        
        prev_params = self.policy_net.get_flat_params()
        success, new_params = linesearch(get_loss, prev_params, fullstep, neggdotstepdir / lm[0])
        self.policy_net.set_flat_params(new_params)
        return loss
        
    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename + '_policy_net.pth')
        torch.save(self.value_net.state_dict(), filename + '_value_net.pth')
        
    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename + '_policy_net.pth'))
        self.value_net.load_state_dict(torch.load(filename + '_value_net.pth')) 
               
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_timesteps', default=30000)
    parser.add_argument('--batch_size', default=50000)
    parser.add_argument('--discount', default=0.1)
    parser.add_argument('--env_name', default='Pendulum-v0')
    parser.add_argument('--gae_lambda', default=0.994)
    parser.add_argument('--max_kl', default=0.001)
    parser.add_argument('--damping', default=0.1)
    parser.add_argument('--l2_reg', default=0.001)
    args = parser.parse_args()
    
    policy_name = 'TRPO'
    

    save_model = False
    load_model = False
    dir_path = os.path.dirname(os.path.realpath(__file__))
    directory_models = dir_path + '/models/'
    directory_results = dir_path + '/results/'
    
    
    seeds = 1000 * np.random.randint(4000, 10000, 5)
    
    
    for seed in seeds:   
        filename_model = directory_models + f"{policy_name}_{args.env_name}_{seed}"
        filename_reward_result = directory_results + \
            f"{'reward'}_{policy_name}_{args.env_name}_{seed}"
        if not os.path.exists(directory_models):
            os.makedirs(directory_models)

        if not os.path.exists(directory_results):
            os.makedirs(directory_results)

        env = gym.make(args.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        running_state = Z_Filter((state_dim, ), clip=5)
        running_reward = Z_Filter((1, ), demean = False, clip=10)

        policy_kwargs = {
            'num_inputs': state_dim,
            'num_outputs': action_dim,
            'max_kl': args.max_kl,
            'tau': args.tau,
            'discount': args.discount,
            'damping': args.damping,
            'l2_reg': args.l2_reg
        }
        
        policy = TRPO(**policy_kwargs)
        
        if load_model:
            policy_file = filename_model
            policy.load(file_name=filename_model)

        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        evaluations_reward = []
        
        rewards = []
        max_reward = 0
        
        for i in count(1):
            
            memory = Memory()
            old_reward = 0
            
            num_steps = 0
            while num_steps < args.batch_size:
                episode_timesteps += 1
                state, done = env.reset(), False
                state = running_state(state)
                
                t = 0
                while not done:
                    t += 1
                    action = policy.select_action(state)
                    action = action.data[0].numpy()                  
                    next_state, reward, done, power, rate = env.step(action)
                    episode_reward += reward                 
                    next_state = running_state(next_state)
                    
                    mask = 1
                    if done:
                        mask = 0
                        
                    memory.push(state, np.array([action]), mask, next_state, reward)
                    state = next_state
                    
                    if done:
                        print("Total T: {}, Episode Num: {}, Episode T: {}, reward: {:.1f}".format(
                            i + 1, episode_num + 1, episode_timesteps +
                            1, episode_reward
                        ))
                        
                        old_reward += episode_reward   

                        evaluations_reward.append(episode_reward)
                        np.save(filename_reward_result, evaluations_reward)

                        episode_num += 1
                        episode_reward = 0
                        episode_timesteps = 0
                        
                num_steps += (t-1)        
                    
            batch = memory.sample()
            policy.train(batch)
                
            
            if i - 1 == 0 or (i - 1 > 0 and old_reward > max_reward): 
                if i - 1 > 0: print("Model is saved: episode: {}, growth: {:.3f}".format(i + 1, old_reward / max_reward))
                max_reward = old_reward
                save_model = True              
            else:
                save_model = False              
            
            if save_model:
                policy.save(filename_model)
                                   
            if episode_num > args.max_timesteps:
                break
            
            
        
        
        
            