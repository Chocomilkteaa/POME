import os
import cv2

from collections import deque
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete, Box

import torch
from torch.utils.tensorboard import SummaryWriter

class ReplayMemory(object):
    def __init__(self, memory_size, number_of_state, discount_rate, device):
        self.states = np.zeros((memory_size, number_of_state), dtype=np.float32)
        self.actions = np.zeros(memory_size, dtype=np.int64)
        self.rewards = np.zeros(memory_size, dtype=np.float32)
        self.next_states = np.zeros((memory_size, number_of_state), dtype=np.float32)
        self.values = np.zeros(memory_size, dtype=np.float32)
        self.log_probs = np.zeros(memory_size, dtype=np.float32)
        self.Q_f = np.zeros(memory_size, dtype=np.float32)
        self.returns = np.zeros(memory_size, dtype=np.float32)

        self.memory_size = memory_size

        self.discount_rate = discount_rate

        self.device = device

        self.start_idx, self.cur_idx = 0, 0

    def push(self, state, action, reward, next_state, value, log_prob):
        assert self.cur_idx < self.memory_size

        self.states[self.cur_idx] = state
        self.actions[self.cur_idx] = action
        self.rewards[self.cur_idx] = reward
        self.next_states[self.cur_idx] = next_state
        self.values[self.cur_idx] = value
        self.log_probs[self.cur_idx] = log_prob

        self.cur_idx += 1

    def done(self, last_value=0):
        self.Q_f[self.cur_idx-1] = self.rewards[self.cur_idx-1] + \
            self.discount_rate * last_value

        R = self.returns[self.cur_idx-1] = last_value

        for i in range(self.cur_idx-2, self.start_idx-1, -1):
            self.Q_f[i] = self.rewards[i] + \
                self.discount_rate * self.values[i+1]

            R = self.returns[i] = self.rewards[i] + R * self.discount_rate

        self.start_idx = self.cur_idx

    def get(self):
        assert self.cur_idx == self.memory_size

        self.start_idx, self.cur_idx = 0, 0

        data = dict(states=torch.as_tensor(self.states, dtype=torch.float32, device=self.device),
                    actions=torch.as_tensor(self.actions, dtype=torch.int64, device=self.device),
                    rewards=torch.as_tensor(self.rewards, dtype=torch.float32, device=self.device),
                    next_states=torch.as_tensor(self.next_states, dtype=torch.float32, device=self.device),
                    values=torch.as_tensor(self.values, dtype=torch.float32, device=self.device),
                    log_probs=torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device),
                    Q_f=torch.as_tensor(self.Q_f, dtype=torch.float32, device=self.device),
                    returns=torch.as_tensor(self.returns, dtype=torch.float32, device=self.device))

        return data

    def reset(self):
        self.start_idx, self.cur_idx = 0, 0

class Actor(torch.nn.Module):
    def __init__(self, number_of_state, number_of_action, hidden_size):
        super(Actor, self).__init__()
        self.layer1 = torch.nn.Linear(number_of_state, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, number_of_action)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.softmax(self.layer3(x), dim=1)
        return x

    def getActionAndProb(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def getProb(self, state, action):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)

        return dist.log_prob(action)

class Critic(torch.nn.Module):
    def __init__(self, number_of_state, hidden_size):
        super(Critic, self).__init__()
        self.layer1 = torch.nn.Linear(number_of_state, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class Reward(torch.nn.Module):
    def __init__(self, number_of_state, hidden_size):
        super(Reward, self).__init__()
        self.layer1 = torch.nn.Linear(number_of_state + 1, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class Transition(torch.nn.Module):
    def __init__(self, number_of_state, hidden_size):
        super(Transition, self).__init__()
        self.layer1 = torch.nn.Linear(number_of_state + 1, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, number_of_state)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
class POME(object):
    def __init__(self, env, size, number_of_state, number_of_action,
                batch_size=128, hidden_size=128, memory_size=10000,
                actor_learning_rate=1e-4, critic_learning_rate=1e-4, reward_learning_rate=1e-4, transition_learning_rate=1e-4,
                max_epsisode=1000, discount_rate=0.99, explore_rate=0.1, clamp_ratio=0.2,
                target_score=500, test_iter=10, log_path='./Log', result_path='./Result', result_name='video', frame_rate=30):
        self.env = env
        self.size = size
        self.number_of_state = number_of_state
        self.number_of_action = number_of_action

        self.batch_size = batch_size

        self.max_epsisode = max_epsisode
        self.discount_rate = discount_rate
        self.explore_rate = explore_rate
        self.clamp_ratio = clamp_ratio

        self.target_score = target_score

        self.test_iter = test_iter

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.writer = SummaryWriter(log_path)
        self.step = 0

        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.result_name = result_name

        self.frame_rate = frame_rate

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.memory_size = memory_size
        self.memory = ReplayMemory(memory_size, number_of_state, discount_rate, self.device)

        self.actor_net = Actor(number_of_state, number_of_action, hidden_size).to(self.device)
        self.critic_net = Critic(number_of_state, hidden_size).to(self.device)
        self.reward_net = Reward(number_of_state, hidden_size).to(self.device)
        self.transition_net = Transition(number_of_state, hidden_size).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_learning_rate)
        self.reward_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=reward_learning_rate)
        self.transition_optimizer = torch.optim.Adam(self.transition_net.parameters(), lr=transition_learning_rate)


    def optimizePolicy(self):
        data = self.memory.get()

        batch_idx = np.arange(self.memory_size)
        np.random.shuffle(batch_idx)

        for i in range(int(self.memory_size / self.batch_size)):
            minibatch_idx = batch_idx[i*self.batch_size:(i+1)*self.batch_size]
            states = data['states'][minibatch_idx]
            actions = data['actions'][minibatch_idx]
            rewards = data['rewards'][minibatch_idx]
            next_states = data['next_states'][minibatch_idx]
            Q_f = data['Q_f'][minibatch_idx]
            returns = data['returns'][minibatch_idx]
            log_probs_old = data['log_probs'][minibatch_idx]

            state_action = torch.cat((states, actions.unsqueeze(1)), dim=1)
            reward_hat = self.reward_net.forward(state_action)
            transition_hat = self.transition_net.forward(state_action)

            Q_b = reward_hat.detach() + self.discount_rate * self.critic_net.forward(transition_hat.detach())
            epsilon = torch.abs(Q_f - Q_b.squeeze())
            epsilon_median = torch.median(epsilon)

            values = self.critic_net.forward(states).squeeze()
            delta = Q_f - values
            delta_pome = delta + self.explore_rate * torch.clamp(epsilon - epsilon_median, -delta, delta)

            log_probs = self.actor_net.getProb(states, actions)
            ratios = torch.exp(log_probs-log_probs_old)
            actor_loss = -torch.min(ratios * delta_pome.detach(), torch.clamp(ratios, 1 - self.clamp_ratio, 1 + self.clamp_ratio) * delta_pome.detach()).mean()
            self.writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.step)

            critic_loss = torch.mean(torch.square((delta_pome + returns.detach() - values)))
            self.writer.add_scalar('critic_loss', critic_loss.item(), global_step=self.step)

            reward_loss = torch.square(torch.mean(torch.abs(rewards - reward_hat.squeeze())))
            self.writer.add_scalar('reward_loss', reward_loss.item(), global_step=self.step)

            transition_loss = torch.sum(torch.square(torch.abs(next_states - transition_hat.squeeze())))
            self.writer.add_scalar('transition_loss', transition_loss.item(), global_step=self.step)

            self.step += 1

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.reward_optimizer.zero_grad()
            reward_loss.backward()
            self.reward_optimizer.step()

            self.transition_optimizer.zero_grad()
            transition_loss.backward()
            self.transition_optimizer.step()

    def getActionAndProb(self, state):
        if isinstance(state, np.ndarray):
          state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.actor_net.getActionAndProb(state)

        return action, log_prob

    def getAction(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.actor_net.getActionAndProb(state)

        return action

    def getValue(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            value = self.critic_net.forward(state)

        return value

    def train(self):
        average_trajectory_reward = deque(maxlen=100)

        for episode in range(self.max_epsisode):
            print(f'Start episode {episode}')
            state, info = self.env.reset(seed = episode)

            rewards = 0

            for j in range(self.memory_size):
                action, log_prob = self.getActionAndProb(state)

                next_state, reward, terminated, truncated, info = env.step(action)

                rewards += reward

                value = self.getValue(state)

                self.memory.push(state, action, reward, next_state, value, log_prob)

                if terminated or truncated or j == self.memory_size-1:
                    average_trajectory_reward.append(rewards)
                    rewards = 0

                    state, info = self.env.reset(seed = episode)

                    if terminated:
                        self.memory.done()
                    else:
                        last_value = self.getValue(next_state)
                        self.memory.done(last_value)
                else:
                    state = next_state

            if np.mean(average_trajectory_reward) >= self.target_score:
                print(f'solved with {episode} epochs')
                break

            self.optimizePolicy()

        self.writer.close()

    def test(self):
        trajectory_rewards = []
        for i in range(self.test_iter):
            state, info = self.env.reset(seed=i)
            trajectory_reward = 0
            trajectory_length = 0

            while True:
                action = self.getAction(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                trajectory_reward += reward
                trajectory_length += 1

                state = next_state

                if terminated or truncated:
                    print(f'Iteration {i}: length:{trajectory_length}, reward: {trajectory_reward}')

                    trajectory_rewards.append(trajectory_reward)

                    break

        print(f'Reward Mean: {np.mean(trajectory_rewards)}, Std: {np.std(trajectory_rewards)}')

        state, info = self.env.reset(seed=np.argmax(trajectory_rewards).item())

        writer = cv2.VideoWriter(os.path.join(self.result_path, f'{self.result_name}.avi'),cv2.VideoWriter_fourcc(*'DIVX'), self.frame_rate, self.size)

        while True:
            action = self.getAction(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            writer.write(self.env.render())

            if terminated or truncated:
                writer.release()

                break

    def pome(self):
        self.train()

        self.test()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def getEnvInfo(env):
    observation, info = env.reset()
    frame = env.render()
    size = (frame.shape[1], frame.shape[0])
    if isinstance(env.observation_space, Box):
        number_of_state = env.observation_space.shape[0]
    elif isinstance(env.observation_space, Discrete):
        number_of_state = env.observation_space.n
    else:
        print('this script only works for Box / Discrete observation spaces.')
        exit()
    if isinstance(env.action_space, Discrete):
        number_of_action = env.action_space.n
    else:
        print('this script only works for Discrete action spaces.')
        exit()

    return size, number_of_state, number_of_action

if __name__ == '__main__':
    set_seed(0)

    env = gym.make("CartPole-v1", render_mode='rgb_array')

    size, number_of_state, number_of_action = getEnvInfo(env)

    alg = POME(env, size, number_of_state, number_of_action, max_epsisode=500, 
               actor_learning_rate=1e-4, critic_learning_rate=1e-4, reward_learning_rate=1e-4, transition_learning_rate=1e-4,
               result_path='./Result/', result_name='pome')
    alg.pome()