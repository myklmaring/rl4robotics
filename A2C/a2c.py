import numpy as np
import random
import time
import gym

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import sys
import argparse
sys.path.insert(1, '/home/michael/Documents/git-repos/baselines')
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


###########################################
parser = argparse.ArgumentParser()
parser.add_argument('--v', type=str, default='1', help='Experiment Number')
opt = parser.parse_args()
exp_name = opt.v

######################################################################


class Plotter():
    def __init__(self):
        self.data = []

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, N_HIDDEN = 10):
        super(ActorCritic, self).__init__()
        self.action_size = action_size

        # Network structure referenced from https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
        self.actor_layer1 = nn.Linear(state_size, N_HIDDEN)
        self.mu = nn.Linear(N_HIDDEN, action_size)
        self.var = nn.Linear(N_HIDDEN, action_size)

        self.critic_layer1 = nn.Linear(state_size, N_HIDDEN)
        self.critic_layer2 = nn.Linear(N_HIDDEN, 1)


    def forward(self, x):
        x = x.to(device)        # double check that input is put to correct (cpu, gpu)
        out = F.relu(self.actor_layer1(x))
        mu = torch.tanh(self.mu(out))
        var = F.softplus(self.var(out)) + 1E-5

        out = F.relu(self.critic_layer1(x))
        value = self.critic_layer2(out)

        return mu, var, value


class A2Ccontinuous:
    def __init__(self, envname, LR, nproc = 16):
        self.nproc = nproc
        self.envname = envname
        self.env = [self.make_env(envname, seed) for seed in range(self.nproc)]
        self.env = SubprocVecEnv(self.env)
        self.model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(device)
        self.optimizer = optim.Adam(self.model.parameters(),LR)

        self.data = {"loss": []}
        self.start_time = None

    def make_env(self, env_id, seed):
        def _f():
            env = gym.make(env_id)
            env.seed(seed)
            return env
        return _f

    def select_action(self, state):
        mus, vars, value = self.model.forward(torch.tensor(state))

        sig = torch.squeeze(torch.sqrt(vars))
        mu = torch.squeeze(mus)
        value = torch.squeeze(value)

        action = torch.normal(mu, sig)
        action = torch.clamp(action, min=self.env.action_space.low[0], max=self.env.action_space.high[0])

        # assuming that each action is independent of each other, and that action probability
        #   distribution is normal, we can model the probability using the probability density
        #   function of the n-dimensional multivariate normal distribution
        log_prob = -torch.log(2*np.pi*torch.prod(sig,1)) - torch.sum((action - mu)**2 / (2*sig**2), 1)

        action = action.detach().cpu().numpy()

        return action, log_prob, value

    def update_a2c(self, rewards, log_probs, values, isdone, state):

        target_vals = []
        t = 0

        # Find the estimated value of the final state of the finite horizon
        _, _, target_val = self.model.forward(torch.tensor(state))
        target_val = torch.squeeze(target_val)

        for reward, done in zip(rewards[::-1], isdone[::-1]):
            target_val += torch.from_numpy(reward).to(device) + torch.from_numpy(1 - done).to(device) \
                          * GAMMA ** t * target_val
            t += 1
            target_vals.append(target_val)

        target_vals = target_vals[::-1]

        loss = 0
        for log_prob, value, target_val in zip(log_probs, values, target_vals):

            advantage = target_val - value
            actor_loss = torch.dot(-log_prob, advantage)
            critic_loss = F.smooth_l1_loss(value, target_val)
            loss += actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # Main training loop.
    def train(self):

        print("Going to be training for a total of {} training steps".format(MAX_TRAINING_STEPS))
        self.start_time = time.time()

        state = self.env.reset()

        for step_num in range(MAX_TRAINING_STEPS):
            # score = 0.0

            rewards = []
            log_probs = []
            values = []
            isdone = []

            for _ in range(FINITE_HORIZON):
                state = state[:, np.newaxis, :]         # allows for batch processing with the NN
                action, log_prob, value = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                # score += reward

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                isdone.append(done)

            # Update Actor - Critic
            self.update_a2c(rewards, log_probs, values, isdone, state[:, np.newaxis, :])

        # np.save('experiments/'+ENV+'/'+ENV+'_total_rewards_'+exp_name+'.npy', total_rewards)
        # np.save('experiments/'+ENV+'/'+ENV+'_mean_rewards_'+exp_name+'.npy', mean_rewards)
        # np.save('experiments/'+ENV+'/'+ENV+'_std_rewards_'+exp_name+'.npy', std_rewards)

        self.env.close()
 
    def test(self, num_episodes, train_episode):
        testing_rewards = []
        for e in range(TEST_EPISODES):
            state = self.env.reset()
            temp_reward = []
            for t in range(MAX_STEPS_PER_EP):
                action, _, _ = self.select_action(state)
                _, reward, done, _ = self.env.step(action)
                temp_reward.append(reward)
                if done:
                    break
            testing_rewards.append(sum(temp_reward))
        return np.mean(testing_rewards), np.std(testing_rewards)


    def demonstrate(self, save_snapshots=None):
        self.env = gym.make(self.envname)
        state = self.env.reset()
        while not done:
            self.env.render()
            action, log_prob, value = self.select_action(state)
            state, reward, done, _ = self.env.step(action)

    def save_experiment(self, environment):

        path = "experiments/" + environment + "_a2c_" + exp_name

        torch.save(self.ActorCritic.state_dict(), path)

        # if you want to load the model, use something similar to the following
        # network = actor()
        # actor.load_state_dict(torch.load(file_path))

        parameters = {
            "Environment Name": self.envname,
            "MAX_EPISODES":MAX_EPISODES,
            "MAX_STEPS_PER_EP":MAX_STEPS_PER_EP,
            "GAMMA":GAMMA,
            "TAU":TAU,
            "LEARNING_RATE_ACTOR":LR_ACTOR,
            "LEARNING_RATE_CRITIC":LR_CRITIC,
        }

        parameters_path = "experiments/" + environment + "_a2c_"+exp_name+".csv"
        with open(parameters_path, "w") as file:
            w = csv.writer(file)
            for key, val in parameters.items():
                w.writerow([key, val, "\n"])

if __name__ == "__main__":
    ############################# PARAMETERS #############################
    # MAX_EPISODES = 5000
    # MAX_STEPS_PER_EP = 300

    MAX_TRAINING_STEPS = 100
    FINITE_HORIZON = 20
    TEST_FREQUENCY = 10
    TEST_EPISODES = 25
    SAVE_FREQUENCY = 100
    GAMMA = 0.9  # discount factor
    LR = 1E-3  # Learning Rate
    N_HIDDEN = 20
    PRINT_DATA = 1  # how often to print data
    RENDER_GAME = False  # View the Episode.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: ", device)

    # ENVIRONMENT = "MountainCarContinuous-v0"
    # ENVIRONMENT = "Pendulum-v0"
    ENVIRONMENT = "LunarLanderContinuous-v2"
    A2C = A2Ccontinuous(ENVIRONMENT, LR)
    A2C.train()
    # A2C.save_experiment(ENVIRONMENT)
