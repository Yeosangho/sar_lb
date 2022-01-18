"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer

from sim_env import sim_env
import subprocess
from subprocess import Popen, PIPE
import random
import time
from torch.nn.utils.rnn import pack_sequence



import torch.optim as optim

from dqn_model import DQN_RAM
from dqn_learn import OptimizerSpec
from utils.schedule import LinearSchedule

BATCH_SIZE = 256
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 3000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 64
TARGER_UPDATE_FREQ = 1000
LEARNING_RATE = 0.0001
ALPHA = 0.95
EPS = 0.01

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}

def dqn_learing(
    q_func,
    optimizer_spec,
    exploration,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000
    ):

    ###############
    # BUILD MODEL #
    ###############

    highscore = 15
    input_arg = 25 #(4+1)*5
    num_actions = 4 # 4 controllers
    controller_num = 4
    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            #print(f"obs.shape {obs.shape}")
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) 
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            with torch.no_grad():
                hidden = torch.zeros(obs.shape, requires_grad = True)
                obs = obs.view(frame_history_len,  25)
                obs = obs.unsqueeze(0)
                actions = model( Variable(obs))
                #print(actions.shape)
                #actions = actions[:,frame_history_len-1,:]
                return actions.data.max(1)[1].cpu().unsqueeze(0)
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function
    device = torch.device('cuda')
    Q = q_func(input_arg, num_actions).type(dtype)
    Q.load_state_dict(torch.load("/scratch/x2026a02/controller_loadbalnacer/281_17.09829503721972_Q.pth"))
    Q.to(device)
    target_Q = q_func(input_arg, num_actions).type(dtype)
    target_Q.load_state_dict(torch.load("/scratch/x2026a02/controller_loadbalnacer/281_17.09829503721972_target_Q.pth"))
    target_Q.to(device)
    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    LOG_EVERY_N_STEPS = 10000
    t = 0
    episode_length = 1000

    for episode in range(10000):
        env = sim_env()
        last_obs = env.reset()
        ep_reward = 0
        #switch_id = env.utilization.index(max(env.utilization))
        utilization_per_controller = []
        for k in range(controller_num):
            utilization_per_controller.append(last_obs[k*5+2]+last_obs[k*5+3] + last_obs[k*5+4])
        #select most overloaded controller
        idx = utilization_per_controller.index(max(utilization_per_controller))
        #get switch list from src ctrl
        switches = env.controller_assigned_info[idx]
        #switch_id = random.choice(switches)
        switch_id = 0
        for itr in range(episode_length):
            t = t+1
            ### Step the env and store the transition
            # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
            # encode_recent_observation will take the latest observation
            # that you pushed into the buffer and compute the corresponding
            # input that should be given to a Q network by appending some
            # previous frames.
    
            # Choose random action if not yet start learning
            last_idx = replay_buffer.store_frame(last_obs)
            recent_observations = replay_buffer.encode_recent_observation()
            random.seed(t)

            actions = select_epilson_greedy_action(Q, recent_observations, t- learning_starts)
            action = actions[0, 0]
            action = action.item()

            # Advance one step
            is_step = False
            is_step, obs, reward = env.step(switch_id, action)
            #switch_id = switch_id +1
            #switch_id = (switch_id % 20)
            #switch_id = env.utilization.index(max(env.utilization))
            utilization_per_controller = []
            for k in range(controller_num):
                utilization_per_controller.append(obs["controllers"][k*5+2]+obs["controllers"][k*5+3] + obs["controllers"][k*5+4])
            #select most overloaded controller
            switch_id = switch_id + 1
            switch_id = (switch_id %20)
            obs["controllers"].extend(obs["switches"][switch_id])
            cid = obs["switches"][switch_id][0]
            obs["controllers"][cid*5+2] = obs["controllers"][cid*5+2] -obs["switches"][switch_id][2]
            obs["controllers"][cid*5+3] = obs["controllers"][cid*5+3] -obs["switches"][switch_id][3]
            obs["controllers"][cid*5+4] = obs["controllers"][cid*5+4] -obs["switches"][switch_id][4]

            
            obs = obs["controllers"]

            #print(obs)
            if(itr == (episode_length-1)):
                done = 1
            else :
                done = 0
            # clip rewards between -1 and 1
            ep_reward = ep_reward + reward
            reward = reward *10
            # Store other info in replay memory
            replay_buffer.store_effect(last_idx, action, reward, done)
            # Resets the environment when reaching an episode boundary.

            last_obs = obs
    
        ### 4. Log progress and keep track of statistics
        episode_rewards = ep_reward

        print(f"{t},{ep_reward}")
    
        #if len(episode_rewards) > 0:
        #    mean_episode_reward = np.mean(episode_rewards[-100:])
        #if len(episode_rewards) > 100:
        #    best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
    
        #Statistic["mean_episode_rewards"].append(mean_episode_reward)
        #Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)
        #print("Timestep %d" % (t,))
        #print("mean reward (100 episodes) %f" % mean_episode_reward)
        #print("best mean reward %f" % best_mean_episode_reward)
        #print("exploration %f" % exploration.value(t))
        sys.stdout.flush()
    
        # Dump statistics to pickle


def main():


    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS, weight_decay=10**-5),
    )

    exploration_schedule = LinearSchedule(1, 0.1)

    dqn_learing(
        q_func=DQN_RAM,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

if __name__ == '__main__':

    main()
    