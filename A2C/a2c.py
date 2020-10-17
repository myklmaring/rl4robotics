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
from tqdm import tqdm
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

    # Network structure referenced from https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, N_HIDDEN = 10):
        super(ActorCritic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size

        self.shared_layer1 = nn.Linear(state_size, N_HIDDEN)
        self.mu = nn.Linear(N_HIDDEN, action_size)
        self.var = nn.Linear(N_HIDDEN, action_size)

        self.critic_layer1 = nn.Linear(N_HIDDEN, 1)


    def forward(self, x):
        x = x.to(self.device)        # double check that input is put to correct (cpu, gpu)
        out = F.relu(self.shared_layer1(x))
        mu = torch.tanh(self.mu(out))
        var = F.softplus(self.var(out)) + 1E-10
        value = self.critic_layer1(out)

        return mu, var, value


class A2Ccontinuous:
    def __init__(self, parameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters = parameters
        self.envname = self.parameters['ENVIRONMENT']
        self.env = [self.make_env(self.envname, seed) for seed in range(self.parameters['n_proc'])]
        self.env = SubprocVecEnv(self.env)
        self.model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.parameters['LR'])

        self.data = {"loss": []}
        self.start_time = None
        self.end_time = None

    def make_env(self, env_id, seed):
        def _f():
            env = gym.make(env_id)
            env.seed(seed)
            return env
        return _f

    def select_action(self, state):
        """
        :param
                    state: numpy array (n_proc x observation_space)
        :return:
                    action:     numpy array  (n_proc x action_space) action selected by model for each environment
                    log_prob:   torch tensor (n_proc x 1) log probability for each action selected
                    value:      torch tensor (n_proc x 1) value assigned to each state by the model
        """

        state = state[:, np.newaxis, :]     # allows for batch processing with the NN
        mu, var, value = self.model.forward(torch.tensor(state))
        value = torch.squeeze(value, dim=1)
        sig = torch.sqrt(var)

        action = torch.normal(mu, sig)
        action = torch.clamp(action, min=self.env.action_space.low[0], max=self.env.action_space.high[0])

        # assuming that each action is independent of each other, and that action probability
        #   distribution is normal, we can model the probability using the probability density
        #   function of the n-dimensional multivariate normal distribution
        log_prob = -torch.log(2*np.pi*torch.prod(sig,2)) - torch.sum((action - mu)**2 / (2*sig**2), 2)

        # This must be numpy to be passed to the openai environments
        action = torch.squeeze(action)
        action = action.detach().cpu().numpy()

        return action, log_prob, value

    def update_a2c(self, rewards, log_probs, values, isdone, state):
        """
        :param log_probs:   torch tensor (n_proc x FINITE_HORIZON) log probability of each action taken at each time and environment
        :param values:      torch tensor (n_proc x FINITE_HORIZON) value of each state at each timepoint and environment
        :param rewards:     list of tensors [n_proc x FINITE_HORIZON] rewards at each timepoint and environment
        :param isdone:      list of tensors [n_proc x FINITE_HORIZON] boolean values representing if each episode is complete
        :param state:       numpy array  (n_proc x observation_space)
        :return:
        """

        # Find the estimated value of the final state of the finite horizon
        state = state[:, np.newaxis, :]     # allows for batch processing with the NN
        _, _, target_val = self.model.forward(torch.tensor(state))
        target_val = torch.squeeze(target_val, dim=2)
        target_vals = []

        for reward, done in zip(rewards[::-1], isdone[::-1]):

            # negative reward because optimization is gradient descent (i.e. lower values are better)
            target_val += -reward + done * self.parameters['GAMMA'] * target_val
            target_vals.append(target_val.clone())

        target_vals = target_vals[::-1]
        target_vals = torch.cat(target_vals, dim=1)

        advantage = target_vals - values
        actor_loss = torch.mean(torch.sum(-log_probs*advantage, 1))
        critic_loss = F.smooth_l1_loss(values, target_vals)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # Main training loop.
    def train(self):

        print("Going to be training for a total of {} training steps".format(self.parameters['MAX_TRAINING_STEPS']))
        self.start_time = time.time()

        state = self.env.reset()

        for step_num in tqdm(range(self.parameters['MAX_TRAINING_STEPS'])):
            # score = 0.0

            rewards = []
            log_probs = []
            values = []
            isdone = []

            for _ in range(self.parameters['FINITE_HORIZON']):
                action, log_prob, value = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                # score += reward

                reward = torch.unsqueeze(torch.tensor(reward), 1).to(self.device)
                done = torch.unsqueeze(torch.tensor(1-done), 1).to(self.device)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                isdone.append(done)

            # format lists into torch tensors
            log_probs = torch.cat(log_probs, dim=1).to(self.device)
            values = torch.cat(values, dim=1).to(self.device)

            # Update Actor - Critic
            self.update_a2c(rewards, log_probs, values, isdone, state)

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
    parameters = {}
    parameters['MAX_TRAINING_STEPS'] = 1000
    parameters['FINITE_HORIZON'] = 20
    parameters['TEST_FREQUENCY'] = 10
    parameters['TEST_EPISODES'] = 25
    parameters['SAVE_FREQUENCY'] = 100
    parameters['GAMMA'] = 0.5
    parameters['entrop_c'] = 1e-4
    parameters['LR'] = 1E-3
    parameters['N_HIDDEN'] = 20
    parameters['n_proc'] = 16
    parameters['PRINT_DATA'] = 1  # how often to print data
    parameters['RENDER_GAME'] = False  # View the Episode.
    parameters['ENVIRONMENT'] = "LunarLanderContinuous-v2"


    print("Using Device: ", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ENVIRONMENT = "MountainCarContinuous-v0"
    # ENVIRONMENT = "Pendulum-v0"

    A2C = A2Ccontinuous(parameters)
    A2C.train()
    # A2C.save_experiment(ENVIRONMENT)
