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

import subprocess
from subprocess import Popen, PIPE
import random
import time
import datetime
import csv
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
    start_time = str(datetime.datetime.now())
    f = open("SDN_RL_" + start_time +".csv", 'w', encoding='utf-8', newline='')
    wr = csv.writer(f, delimiter=',')
    wr.writerow(["TimeStep","CPU_UTIL_DEVIATION", "RAM_UTIL_DEVIATION", "NET_UTIL_DEVIATION", "MEAN_REWARD", "MAX_REWARD" ,"Exploration"])

    ###############
    # BUILD MODEL #
    ###############

    # 4 * 5(CID, SID, CPU, RAM, PKT)
    input_arg = frame_history_len * 20
    num_actions = 3

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            #print(f"obs.shape {obs.shape}")
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) 
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            with torch.no_grad():
                return model(Variable(obs)).data.max(1)[1].cpu().unsqueeze(0)
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function
    Q = q_func(input_arg, num_actions).type(dtype)
    target_Q = q_func(input_arg, num_actions).type(dtype)

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
    t = 0
        
    time.sleep(6)
    last_obs = [0] * 20
    ep_reward = []
    switch_bw_threshold = 0.1
    switch_overhead = [1.0]
    switch_id = 0
    for itr in range(10000):
        t = t+1
        ### Step the env and store the transition
        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
        # encode_recent_observation will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        action_done = False
        success = False
        while(action_done is False):
            # Choose random action if not yet start learning
            if(np.max(switch_overhead) > switch_bw_threshold):
                last_idx = replay_buffer.store_frame(last_obs)
                recent_observations = replay_buffer.encode_recent_observation()
    
                if t > learning_starts:
                    actions = select_epilson_greedy_action(Q, recent_observations, t- learning_starts)
                    #print(f"actions {actions} actions.shpae : {actions.shape}")
                    action = actions[0, 0]
                else:
                    action = random.randrange(num_actions)
                # Advance one step
                ###########################################################################################
                response = requests.get("http://210.107.197.219:9200/step/" + str(switch_id) + "/" + str(controller_id))
                success = response.success
                if(success)
                    state = response.state  
                    reward = response.reward
                    ###########################################################################################
                    done = 0
                    # clip rewards between -1 and 1
                    ep_reward.append(reward)
                    # Store other info in replay memory
                    replay_buffer.store_effect(last_idx, action, reward, done)
                    # Resets the environment when reaching an episode boundary.
            
                    last_obs = obs
                    
                    action_done = True
                    ### Perform experience replay and train the network.
                    # Note that this is only done if the replay buffer contains enough samples
                    # for us to learn something useful -- until then, the model will not be
                    # initialized and random actions should be taken
            switch_id = switch_id + 1
            switch_id = (switch_id % 16)
            #현재 스위치의 부하 확인 
            #CPU STD DEV 계산
            #RAM STD DEV 계산
            #NET STD DEV 계산



        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            # Convert numpy nd_array to torch variables for calculation
            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype))
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)
    
            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()
    
            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.
            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1))
            #print(current_Q_values.shape)
            #print(act_batch.shape)

            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            next_max_q = target_Q(next_obs_batch).detach().max(1)[0]
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            #print(f"next Q values {next_Q_values.shape}")
            target_Q_values = rew_batch + (gamma * next_Q_values)
            # Compute Bellman error
            #print(target_Q_values.shape)
            bellman_error = target_Q_values.unsqueeze(1) - current_Q_values
            #print(bellman_error.shape)

            # clip the bellman error between [-1 , 1]
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            # Note: clipped_bellman_delta * -1 will be right gradient
            d_error = clipped_bellman_error * -1.0
            # Clear previous gradients before backward pass
            optimizer.zero_grad()
            # run backward pass
            current_Q_values.backward(d_error.data)
    
            # Perfom the update
            optimizer.step()
            num_param_updates += 1
    
            # Periodically update the target network by Q network to target Q network
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())
        
        if(t % 100 == 0):
            wr.writerow([t,"CPU_UTIL_DEVIATION", "RAM_UTIL_DEVIATION", "NET_UTIL_DEVIATION", np.mean(ep_reward), np.mean(ep_reward), exploration.value(t)])
            ep_reward = [] 
    wr.close()
    