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
    target_update_freq=10000,
    times=1
    ):

    ###############
    # BUILD MODEL #
    ###############


    controller_num = num_actions = 4 * 1
    

    switch_num = 20 * times
    input_arg = (controller_num +1) * 5  #(4+1)*5

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
                obs = obs.view(frame_history_len,  input_arg)
                obs = obs.unsqueeze(0)
                actions = model( Variable(obs))
                #print(actions.shape)
                #actions = actions[:,frame_history_len-1,:]
                return actions.data.max(1)[1].cpu().unsqueeze(0)
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function
    Q = q_func(input_arg, num_actions).type(dtype)
    target_Q = q_func(input_arg, num_actions).type(dtype)

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, input_arg)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    LOG_EVERY_N_STEPS = 10000
    t = 0
    episode_length = 300

    for episode in range(10000):
        env = sim_env(switch_num, controller_num)
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
        switch_id = random.choice(switches)

        switch_nb = env.nbs[switch_id]
        switch_cpu = env.cpus[switch_id]
        switch_ram = env.rams[switch_id]
        last_obs.extend([idx, 1, switch_nb, switch_cpu, switch_ram])
        last_obs[idx*5+2] = last_obs[idx*5+2] -switch_nb
        last_obs[idx*5+3] = last_obs[idx*5+3] -switch_cpu
        last_obs[idx*5+4] = last_obs[idx*5+4] -switch_ram
        #print(last_obs[0])


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

            if t > learning_starts:
                #print(recent_observations[0])
                #print(recent_observations[1])
                actions = select_epilson_greedy_action(Q, recent_observations, t- learning_starts)
                #print(f"actions {actions} actions.shpae : {actions.shape}")
                action = actions[0, 0]
                action = action.item()
            else:
                action = random.randrange(num_actions)
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
            idx = utilization_per_controller.index(max(utilization_per_controller))
            #get switch list from src ctrl
            switches = env.controller_assigned_info[idx]
            switch_id = random.choice(switches)

            obs["controllers"].extend(obs["switches"][switch_id])
            obs["controllers"][idx*5+2] = obs["controllers"][idx*5+2] -obs["switches"][switch_id][2]
            obs["controllers"][idx*5+3] = obs["controllers"][idx*5+3] -obs["switches"][switch_id][3]
            obs["controllers"][idx*5+4] = obs["controllers"][idx*5+4] -obs["switches"][switch_id][4]

            
            obs = obs["controllers"]

            #print(obs)
            if(itr == (episode_length-1)):
                done = 1
            else :
                done = 0
            # clip rewards between -1 and 1
            ep_reward = ep_reward + reward
            reward = reward *10
            reward = max(-1.0, min(reward, 1.0))
            # Store other info in replay memory
            replay_buffer.store_effect(last_idx, action, reward, done)
            # Resets the environment when reaching an episode boundary.
            last_obs = obs.copy()

            ### Perform experience replay and train the network.
            # Note that this is only done if the replay buffer contains enough samples
            # for us to learn something useful -- until then, the model will not be
            # initialized and random actions should be taken
            if (t > learning_starts and
                    t % learning_freq == 0 and
                    replay_buffer.can_sample(batch_size)):
                # Use the replay buffer to sample a batch of transitions
                # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
                # in which case there is no Q-value at the next state; at the end of an
                # episode, only the current state reward contributes to the target
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

                #print(obs_batch[0][0])
                ##print(obs_batch[0][1])
                #print(obs_batch[1][0])
                #exit()
                # Convert numpy nd_array to torch variables for calculation
                obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
                act_batch = Variable(torch.from_numpy(act_batch).long())
                rew_batch = Variable(torch.from_numpy(rew_batch))
                next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype))
                not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)
                #print(next_obs_batch)
                if USE_CUDA:
                    act_batch = act_batch.cuda()
                    rew_batch = rew_batch.cuda()
    
                # Compute current Q value, q_func takes only state and output value for every state-action pair
                # We choose Q based on action taken.
                hidden = torch.zeros(obs_batch.shape, requires_grad = True)
                #obs_batch =pack_sequence(obs_batch)
                #print(obs_batch.shape)
                obs_batch = obs_batch.view(batch_size,frame_history_len,  input_arg).requires_grad_()
                #obs_batch = obs_batch.view(obs_batch.size(0),1, obs_batch.size(1))
                #print(obs_batch[0][0])
                qval = Q(obs_batch)
                #print(qval.shape)
                #qval = qval[:,frame_history_len-1,:]

                current_Q_values = qval.gather(1, act_batch.unsqueeze(1))
                
                #print(current_Q_values.shape)
                #print(act_batch.shape)
                # Compute next Q value based on which action gives max Q values
                # Detach variable from the current graph since we don't want gradients for next Q to propagated
                hidden2 = torch.zeros(obs_batch.shape, requires_grad = True)
                #next_obs_batch = torch.unsqueeze(next_obs_batch, 0)
                next_obs_batch = next_obs_batch.view(batch_size,frame_history_len,  input_arg).requires_grad_()

                target_qval = target_Q(next_obs_batch)
                #target_qval = target_qval[:,frame_history_len-1,:]
                next_max_q = target_qval.detach().max(1)[0]
                
                
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
    
        ### 4. Log progress and keep track of statistics
        episode_rewards = ep_reward
        with open(f'/scratch/x2026a02/stochastic_controller_loadbalnacer/_{switch_num}_{controller_num}_stochastic_mode.csv', '+a') as f:
            print('{ep}, {score}'
                .format(ep=episode,score=ep_reward),
                file=f
                )

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
        if(episode % 1000 == 0):
            # Dump statistics to pickle
            torch.save(Q.state_dict(), f"/scratch/x2026a02/stochastic_controller_loadbalnacer/{episode}_{switch_num}_{controller_num}_Q.pth")
            torch.save(target_Q.state_dict(), f"/scratch/x2026a02/stochastic_controller_loadbalnacer/{episode}_{switch_num}_{controller_num}_target_Q.pth")
    # Dump statistics to pickle
    torch.save(Q.state_dict(), f"/scratch/x2026a02/stochastic_controller_loadbalnacer/{episode}_Q.pth")
    torch.save(target_Q.state_dict(), f"/scratch/x2026a02/stochastic_controller_loadbalnacer/{episode}_{switch_num}_{controller_num}_target_Q.pth")
    