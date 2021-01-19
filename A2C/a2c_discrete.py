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
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import ffmpeg

######################################################################
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, N_HIDDEN=64):
        super(ActorCritic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.state_size = state_size

        self.shared_layer1 = nn.Linear(state_size, N_HIDDEN)
        self.shared_layer2 = nn.Linear(N_HIDDEN, N_HIDDEN)
        self.actions = nn.Linear(N_HIDDEN, action_size)

        self.value = nn.Linear(N_HIDDEN, 1)

    def forward(self, x):
        x = x.to(self.device)
        out = F.relu(self.shared_layer1(x))
        out = F.relu(self.shared_layer2(out))
        actions = F.softmax(self.actions(out), dim=2)

        value = self.value(out)

        return actions, value

class A2C:
    def __init__(self, parameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters = parameters
        self.envname = self.parameters['ENVIRONMENT']
        self.env = gym.vector.make(self.envname, num_envs=self.parameters['N_PROC'])

        # observation space is Box(x,y,z) i.e. continuous
        # Shape method of Discrete(n) returned (), n method returns n from Discrete(n)
        # for vectorized environment, env.observation_space returns (n_proc, observation_space)
        #                             env.action space is Tuple(Discrete(n), Discrete(n), ..., Discrete(n))

        self.model = ActorCritic(self.env.observation_space.shape[1], self.env.action_space[0].n,
            N_HIDDEN=self.parameters['N_HIDDEN']).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameters['LR'])
        self.start_time = None
        self.end_time = None
        self.best_network = None
        self.best_testmean = -np.inf

    def select_action(self, state, model):
        """
        :param
                    state: numpy array (N_PROC x observation_space)
        :return:
                    action:     numpy array  (N_PROC x action_space) action selected by model for each environment
                    log_prob:   torch tensor (N_PROC x 1) log probability for each action selected
                    value:      torch tensor (N_PROC x 1) value assigned to each state by the model
                    entropy:    torch scalar () average entropy over all samples
        """
        state = state[:, np.newaxis, :]  # allows for batch processing with the NN
        action_distribution, value = model(torch.tensor(state).float())
        value = torch.squeeze(value, dim=1)

        distribution = torch.distributions.Categorical(action_distribution)
        action = distribution.sample()
        # action = torch.clamp(action, min=self.env.action_space.low[0], max=self.env.action_space.high[0])
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        # This must be numpy to be passed to the openai environments
        action = torch.squeeze(action, 1)
        action = action.detach().cpu().numpy()

        return action, log_prob, value, entropy

    def update_a2c(self, rewards, log_probs, values, isdone, state, entropies):
        """
        :param log_probs:   torch tensor (N_PROC x FINITE_HORIZON) log probability of each action taken at each time and environment
        :param values:      torch tensor (N_PROC x FINITE_HORIZON) value of each state at each timepoint and environment
        :param rewards:     list of tensors [N_PROC x FINITE_HORIZON] rewards at each timepoint and environment
        :param isdone:      list of tensors [N_PROC x FINITE_HORIZON] boolean values representing if each episode is complete
        :param state:       numpy array  (N_PROC x observation_space)
        :param entropies    torch tensor (N_PROC, )

        :return: loss:      numpy scalar (scalar) loss used for backpropagation
        """

        # Find the estimated value of the final state of the finite horizon
        state = state[:, np.newaxis, :]  # allows for batch processing with the NN
        _, td_target = self.model(torch.tensor(state).float())
        td_target = torch.squeeze(td_target, dim=2)
        td_targets = []

        for reward, done in zip(rewards[::-1], isdone[::-1]):
            td_target = reward + done * self.parameters['GAMMA'] * td_target
            td_targets.append(td_target)

        td_targets = torch.cat(td_targets[::-1], dim=1)

        advantage = td_targets - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = F.smooth_l1_loss(td_targets, values)
        entropy_loss = self.parameters['ENTROPY_C'] * entropies.mean()

        # print("actor loss:", actor_loss.clone().detach().cpu().numpy())
        # print("critic loss:", critic_loss.clone().detach().cpu().numpy())
        # print("entropy loss:", entropy_loss.clone().detach().cpu().numpy())

        loss = actor_loss + critic_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

        return loss.clone().detach().cpu().numpy()

    # Main training loop.
    def train(self):

        print("Going to be training for a total of {} training steps".format(self.parameters['MAX_TRAINING_STEPS']))
        self.start_time = time.time()

        state = self.env.reset()
        loss_list, test_list = [], []

        for step_num in tqdm(range(self.parameters['MAX_TRAINING_STEPS'])):

            rewards, log_probs, values, isdone, entropies = [], [], [], [], []

            for _ in range(self.parameters['FINITE_HORIZON']):
                action, log_prob, value, entropy = self.select_action(state, self.model)
                state, reward, done, _ = self.env.step(action)

                reward = torch.unsqueeze(torch.tensor(reward), 1).to(self.device)
                done = torch.unsqueeze(torch.tensor(1 - done), 1).to(self.device)

                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)

                rewards.append(reward)
                isdone.append(done)

            # format lists into torch tensors
            log_probs = torch.cat(log_probs, dim=1).to(self.device)
            values = torch.cat(values, dim=1).to(self.device)
            entropies = torch.cat(entropies).to(self.device)

            # Update Actor-Critic
            loss = self.update_a2c(rewards, log_probs, values, isdone, state, entropies)
            loss_list.append(loss)

            if (step_num % self.parameters['PRINT_DATA']) == 0 and step_num != 0:
                y = np.array(loss_list)
                kernel = (1 / self.parameters['PRINT_DATA']) * np.ones(self.parameters['PRINT_DATA'])
                ma_y = np.convolve(y, kernel, mode='same')
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                plt.plot(y, '-b')
                plt.plot(ma_y, '-r')
                plt.axhline(color='k')
                plt.xlabel("Number of Training Steps")
                plt.ylabel("Loss")
                plt.title("Training Loss")
                plt.legend(['Loss', 'Moving Average (n={})'.format(self.parameters['PRINT_DATA'])])
                plt.savefig("train_loss.png")
                plt.close()

            if (step_num % self.parameters['TEST_FREQUENCY']) == 0 and step_num != 0:
                test_mean, test_std = self.test()
                test_list.append([test_mean, test_std])
                x = np.arange(1, step_num, self.parameters['TEST_FREQUENCY'])
                y = np.array(test_list)
                plt.errorbar(x, y[:, 0], yerr=y[:, 1], fmt='.k')
                plt.axhline(color='k')
                plt.xlabel("Number of Training Steps")
                plt.ylabel("Mean Episode Cumulative Reward (n={})".format(self.parameters['TEST_EPISODES']))
                plt.title("Test Episode Cumulative Reward Progression")
                plt.savefig("test_reward.png")
                plt.close()

                # save the network from the game with the best test mean reward
                if test_mean > self.best_testmean:
                    self.best_network = self.model
                    self.best_testmean = test_mean

        self.env.close()

    def test(self):
        test_env = gym.make(self.envname)
        testing_rewards = []
        for _ in range(self.parameters['TEST_EPISODES']):
            state = test_env.reset()
            temp_reward = 0
            for _ in range(self.parameters['MAX_STEPS_PER_EP']):
                action, _, _, _ = self.select_action(state[None, :], self.model)
                state, reward, done, _ = test_env.step(action[0])
                temp_reward += reward
                if done:
                    break
            testing_rewards.append(temp_reward)
        return np.mean(testing_rewards), np.std(testing_rewards)

    def demonstrate(self):
        env = gym.make(self.envname)
        state = env.reset()
        total_reward = 0
        done = False
        images = []

        while not done:
            images.append(env.render(mode='rgb_array'))
            action, _, _, _ = self.select_action(state[None, :], model=self.best_network)
            state, reward, done, _ = env.step(action[0])
            total_reward += reward

        file = '/home/michael/Documents/git-repos/rl4robotics/A2C/movie.mp4'
        self.vidwrite(file, images)
        env.close()

    # referenced from Kyle Mcdonald on https://github.com/kkroening/ffmpeg-python/issues/246
    def vidwrite(self, fn, images, framerate=60, vcodec='libx264'):
        if not isinstance(images, np.ndarray):
            images = np.asarray(images)
        n,height,width,channels = images.shape
        print(images.shape)
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .global_args('-loglevel', 'warning')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        for frame in images:
            process.stdin.write(
                frame
                .astype(np.uint8)
                .tobytes()
            )
        process.stdin.close()
        process.wait()

    def save_experiment(self, environment):

        path = "experiments/" + environment + "_a2c_" + exp_name

        torch.save(self.ActorCritic.state_dict(), path)

        # if you want to load the model, use something similar to the following
        # network = actor()
        # actor.load_state_dict(torch.load(file_path))

        parameters = {
            "Environment Name": self.envname,
            "MAX_EPISODES": MAX_EPISODES,
            "MAX_STEPS_PER_EP": MAX_STEPS_PER_EP,
            "GAMMA": GAMMA,
            "TAU": TAU,
            "LEARNING_RATE_ACTOR": LR_ACTOR,
            "LEARNING_RATE_CRITIC": LR_CRITIC,
        }

        parameters_path = "experiments/" + environment + "_a2c_" + exp_name + ".csv"
        with open(parameters_path, "w") as file:
            w = csv.writer(file)
            for key, val in parameters.items():
                w.writerow([key, val, "\n"])


if __name__ == "__main__":
    ############################# PARAMETERS #############################
    parameters = {}
    parameters['MAX_TRAINING_STEPS'] = 10000
    parameters['FINITE_HORIZON'] = 5
    parameters['N_PROC'] = 3
    parameters['PRINT_DATA'] = 500  # how often to print data
    parameters['TEST_FREQUENCY'] = 500
    parameters['TEST_EPISODES'] = 5
    parameters['MAX_STEPS_PER_EP'] = 500
    parameters['SAVE_FREQUENCY'] = 100
    parameters['GAMMA'] = 0.98
    parameters['ENTROPY_C'] = 0
    parameters['LR'] = 1E-3
    parameters['N_HIDDEN'] = 64
    parameters['ENVIRONMENT'] = "CartPole-v1"

    print("Using Device: ", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Continuous Game Environments
    # "LunarLanderContinuous-v2"
    # "MountainCarContinuous-v0"
    # "Pendulum-v0"

    # Discrete Game Environments
    # "CartPole-v1"
    # MountainCar-v0


    A2C = A2C(parameters)
    A2C.train()
    A2C.demonstrate()
    # A2C.save_experiment(ENVIRONMENT)
