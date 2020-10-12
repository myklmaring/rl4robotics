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
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


############################# PARAMETERS #############################
# MAX_EPISODES = 5000
# MAX_STEPS_PER_EP = 300

MAX_TRAINING_STEPS = 5000
FINITE_HORIZON = 20
TEST_FREQUENCY = 10
TEST_EPISODES = 25
SAVE_FREQUENCY = 100
GAMMA = 0.9           # discount factor
LR = 1E-3             # Learning Rate
N_HIDDEN = 128
PRINT_DATA = 1        # how often to print data
RENDER_GAME = False   # View the Episode. 

# ENVIRONMENT = "MountainCarContinuous-v0"
# ENV = 'MountainCar'
#ENVIRONMENT = "Pendulum-v0"
# ENV = 'Pendulum'
ENVIRONMENT = "LunarLanderContinuous-v2"
ENV = 'LunarLander'


###########################################
parser = argparse.ArgumentParser()
parser.add_argument('--v', type=str, default='1', help='Experiment Number')
opt = parser.parse_args()
exp_name = opt.v

######################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device)

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
        out = F.relu(self.actor_layer1(x))
        mu = torch.tanh(self.mu(out))
        var = F.softplus(self.var(out)) + 1E-5

        out = F.relu(self.critic_layer1(x))
        value = self.critic_layer2(out)

        return mu, var, value


class A2Ccontinuous:
    def __init__(self, envname, LR):
        
        self.envname = envname
        self.env = [self.make_env(env_name, seed) for seed in range(nproc)]
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
        var = vars.data.cpu().numpy()

        sig = np.squeeze(np.sqrt(var))
        mu = np.squeeze(mus.data.cpu().numpy())
        value = np.squeeze(value.data.cpu().numpy())

        action = np.random.normal(mu, sig)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        log_prob = -np.log(np.sqrt(2*np.pi*sig)) - (action - mu)**2 / (2*sig**2)

        return action, log_prob, value

    def update_a2c(self, rewards, log_probs, values, isdone, state):

        target_vals = []
        t = 0

        # Find the estimated value of the final state of the finite horizon
        _, _, target_val = self.model.forward(torch.tensor(state))

        for reward, done in zip(rewards[::-1], isdone[::-1]):
            target_val += (1 - done) * GAMMA ** t * target_val + reward
            t += 1
            target_vals.append(target_val)

        target_vals = target_vals[::-1]
        target_vals = torch.tensor(target_vals)

        loss = 0
        for log_prob, value, target_val in zip(log_probs, values, target_vals):

            advantage = target_val - value
            actor_loss = -log_prob * advantage
            critic_loss = F.smooth_l1_loss(value, target_val)
            loss += critic_loss + actor_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # Main training loop.
    def train(self):

        print("Going to be training for a total of {} training steps".format(MAX_TRAINING_STEPS))
        self.start_time = time.time()
        # for e in range(MAX_EPISODES):
        #     state = self.env.reset()
        #     score = 0.0
        #     step_num = 0
        #
        #     rewards = []
        #     log_probs = []
        #     values = []
        #
        #     for t in range(MAX_STEPS_PER_EP):
        #
        #         step_num += 1
        #
        #         if RENDER_GAME and (e+1) % 25 == 0:
        #             self.env.render()
        #
        #         state = self.scale_state(state)
        #         action, log_prob, value = self.select_action(state)
        #         state, reward, done, _ = self.env.step(action)
        #         score += reward
        #         rewards.append(reward)
        #         values.append(value)
        #         log_probs.append(log_prob)
        #         if done:
        #             break
        #
        #     total_rewards.append(score)
        #
        #      # Update Actor - Critic
        #     self.update_a2c(rewards, log_probs, values)
        #
        #     if (e+1) % PRINT_DATA == 0:
        #         print("Episode: {}, reward: {}, steps: {}".format(e+1, total_rewards[e], step_num))
        #
        #     if (e+1) % TEST_FREQUENCY == 0:
        #         print("-"*10 + " testing now " + "-"*10)
        #         mean_reward, std_reward = self.test(TEST_EPISODES,e)
        #         print('Mean Reward Achieved : {} \nStandard Deviation : {}'.format(mean_reward, std_reward))
        #         mean_rewards.append(mean_reward)
        #         std_rewards.append(std_reward)
        #         print("-"*50)

        step_num = 0
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
            self.update_a2c(rewards, log_probs, values, isdone, state)

        np.save('experiments/'+ENV+'/'+ENV+'_total_rewards_'+exp_name+'.npy', total_rewards)
        np.save('experiments/'+ENV+'/'+ENV+'_mean_rewards_'+exp_name+'.npy', mean_rewards)
        np.save('experiments/'+ENV+'/'+ENV+'_std_rewards_'+exp_name+'.npy', std_rewards)

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
    A2C = A2Ccontinuous(ENVIRONMENT, LR)
    A2C.train()
    # A2C.save_experiment(ENVIRONMENT)
