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
import copy
import math
import csv
import torch.optim as optim
from dqn_model import DQN_RAM
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

    num_actions = 3
    controller_num = times * 4
    

    switch_num = 20 * times
    input_arg = controller_num

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        #sample = random.random()
        #eps_threshold = exploration.value(t)
        #print(eps_threshold)
        if sample > eps_threshold:
            #print(f"obs.shape {obs.shape}")
            
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            with torch.no_grad():
                obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)
                obs = obs.view(frame_history_len,  input_arg)
                obs = obs.unsqueeze(0)
                
                actions = model(obs)
                #print(actions)
                #print(actions.shape)
                #actions = actions[:,frame_history_len-1,:]
                return actions.data.max(1)[1].cpu().unsqueeze(0)
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])
    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action_val(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        #print(eps_threshold)           
        # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
        with torch.no_grad():
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)

            obs = obs.view(frame_history_len,  input_arg)
            obs = obs.unsqueeze(0)
            #print(obs[0][31])
            actions = model(obs)
            print(actions)
            #print(actions)
            #print(actions)
            #print(actions)
            #print(actions.shape)
            #actions = actions[:,frame_history_len-1,:]
            return actions.data.max(1)[1].cpu().unsqueeze(0)

    # Initialize target q function and q function
    #Qs = []
    #targetQs = []
    #for i in range(controller_num):
    #    Q = DQN_RAM(input_arg, num_actions).type(dtype)
    #    Qs.append(Q)
    #for i in range(controller_num):
    #    target_Q = DQN_RAM(input_arg, num_actions).type(dtype)
    #    targetQs.append(target_Q)
#
    ## Construct Q network optimizer function
    #optimizers = []
    #for i in range(controller_num):
    #    optimizer = optimizer_spec.constructor(Qs[i].parameters(), **optimizer_spec.kwargs)
    #    optimizers.append(optimizer)
#
    #replay_buffers = []
    #for i in range(controller_num):
    #    # Construct the replay buffer
    #    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, input_arg)
    #    replay_buffers.append(replay_buffer)
    Q = DQN_RAM(input_arg, num_actions).type(dtype)
    target_Q = DQN_RAM(input_arg, num_actions).type(dtype)
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, input_arg)
    ###############
    # RUN ENV     #
    ###############
    num_param_updates_per_controller = []
    for i in range(controller_num):
        num_param_updates = 0
        num_param_updates_per_controller.append(num_param_updates)
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    LOG_EVERY_N_STEPS = 10000
    ts = []
    for i in range(controller_num):
        t = 0
        ts.append(t)
    episode_length = 300

    for episode in range(10000):
        env = sim_env(switch_num, controller_num)
        controller_state = env.reset()

        ep_reward = 0
        # pick controller with utilization roulettes
        utilization_per_controller = []
        for k in range(controller_num):
            utilization_per_controller.append(controller_state[k*5+2]+controller_state[k*5+3] + controller_state[k*5+4])
        last_obs = copy.deepcopy(utilization_per_controller)

        sum_utilization = sum(utilization_per_controller)
        utilization_per_controller = utilization_per_controller / sum_utilization
        target_controller_id = np.random.choice(list(range(len(utilization_per_controller))), size=1, p=utilization_per_controller)   
        target_controller_id = target_controller_id[0]


        for itr in range(episode_length):
            #print(target_controller_id)
            ts[target_controller_id] = ts[target_controller_id]+1
            
            ### Step the env and store the transition
            # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
            # encode_recent_observation will take the latest observation
            # that you pushed into the buffer and compute the corresponding
            # input that should be given to a Q network by appending some
            # previous frames.
    
            # Choose random action if not yet start learning
            last_idx =0
            for i in range(controller_num):
                    if(i == target_controller_id):
                        last_idx = replay_buffers[i].store_frame(last_obs)
                    else:
                        replay_buffers[i].store_frame(last_obs)
            recent_observations = replay_buffers[target_controller_id].encode_recent_observation()
            random.seed(ts[target_controller_id])

            if ts[target_controller_id] > learning_starts:
                #print(recent_observations[0])
                #print(recent_observations[1])
                #
                #print(ts[target_controller_id])
                actions = select_epilson_greedy_action(Qs[target_controller_id], recent_observations, ts[target_controller_id] - learning_starts)
                #print(f"actions {actions} actions.shpae : {actions.shape}")
                action = actions[0, 0]
                action = action.item()
                #print(action)
            else:
                action = random.randrange(num_actions)

            # if action stay(0) / import(1) /export(2)
            dest_ctrl_id =0
            target_switch_id = 0
            if(action == 0) : 
                target_switch_id = -1
                dest_ctrl_id = -1
            elif(action == 1) :
                dest_ctrl_id = target_controller_id
                nb_per_controller = []
                cpu_per_controller = []
                ram_per_controller = []
                utilization_per_controller = []
                for k in range(controller_num):
                    nb_per_controller.append(controller_state[k*5+2])
                    cpu_per_controller.append(controller_state[k*5+3])
                    ram_per_controller.append(controller_state[k*5+4])
                    utilization_per_controller.append(controller_state[k*5+2]+controller_state[k*5+3] + controller_state[k*5+4])   
                src_ctrl_id = utilization_per_controller.index(max(utilization_per_controller))
                switches_in_src = env.controller_assigned_info[src_ctrl_id]
                util_gap_for_switches_in_src = []
                for switch_id in switches_in_src :
                    util_gap = math.pow(nb_per_controller[src_ctrl_id] - nb_per_controller[dest_ctrl_id] - env.nbs[switch_id],2) +math.pow(cpu_per_controller[src_ctrl_id] - cpu_per_controller[dest_ctrl_id] - env.cpus[switch_id],2)  +math.pow(ram_per_controller[src_ctrl_id] - ram_per_controller[dest_ctrl_id] - env.rams[switch_id],2)
                    util_gap_for_switches_in_src.append(util_gap)
                target_switch_id = switches_in_src[util_gap_for_switches_in_src.index(min(util_gap_for_switches_in_src))]   
            
            else :
                src_ctrl_id = target_controller_id
                nb_per_controller = []
                cpu_per_controller = []
                ram_per_controller = []
                utilization_per_controller = []
                for k in range(controller_num):
                    nb_per_controller.append(controller_state[k*5+2])
                    cpu_per_controller.append(controller_state[k*5+3])
                    ram_per_controller.append(controller_state[k*5+4])
                    utilization_per_controller.append(controller_state[k*5+2]+controller_state[k*5+3] + controller_state[k*5+4])
                #print(utilization_per_controller)   
                dest_ctrl_id = utilization_per_controller.index(min(utilization_per_controller))
                #print(dest_ctrl_id)
                switches_in_src = env.controller_assigned_info[src_ctrl_id]
                util_gap_for_switches_in_src = []
                for switch_id in switches_in_src :
                    util_gap = math.pow(nb_per_controller[src_ctrl_id] - nb_per_controller[dest_ctrl_id] - env.nbs[switch_id],2) +math.pow(cpu_per_controller[src_ctrl_id] - cpu_per_controller[dest_ctrl_id] - env.cpus[switch_id],2)  +math.pow(ram_per_controller[src_ctrl_id] - ram_per_controller[dest_ctrl_id] - env.rams[switch_id],2)
                    util_gap_for_switches_in_src.append(util_gap)
                target_switch_id = switches_in_src[util_gap_for_switches_in_src.index(min(util_gap_for_switches_in_src))]    
        

            # Advance one step
            is_step = False
            is_step, controller_state, reward = env.step(target_switch_id, dest_ctrl_id)
            controller_state = controller_state["controllers"]
            utilization_per_controller = []
            for k in range(controller_num):
                utilization_per_controller.append(controller_state[k*5+2]+controller_state[k*5+3] + controller_state[k*5+4])
            obs = copy.deepcopy(utilization_per_controller)
            #print(obs)
            sum_utilization = sum(utilization_per_controller)
            utilization_per_controller = utilization_per_controller / sum_utilization
            next_target_controller_id = np.random.choice(list(range(len(utilization_per_controller))), size=1, p=utilization_per_controller)   

            #print(obs)
            if(itr == (episode_length-1)):
                done = 1
            else :
                done = 0
            # clip rewards between -1 and 1
            ep_reward = ep_reward + reward
            reward = reward *10
            #print(reward)
            reward = max(-1.0, min(reward, 1.0))
            # Store other info in replay memory
            replay_buffers[target_controller_id].store_effect(last_idx, action, reward, done)
            # Resets the environment when reaching an episode boundary.
            last_obs = obs.copy()
            

            ### Perform experience replay and train the network.
            # Note that this is only done if the replay buffer contains enough samples
            # for us to learn something useful -- until then, the model will not be
            # initialized and random actions should be taken
            
            if (ts[target_controller_id] > learning_starts and
                    ts[target_controller_id] % learning_freq == 0 and
                    replay_buffers[target_controller_id].can_sample(batch_size)):
                # Use the replay buffer to sample a batch of transitions
                # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
                # in which case there is no Q-value at the next state; at the end of an
                # episode, only the current state reward contributes to the target
                #print(target_controller_id)
                Qs[target_controller_id].train()
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffers[target_controller_id].sample(batch_size)
    
                #print(obs_batch[0][0])
                ##print(obs_batch[0][1])
                #print(obs_batch[1][0])
                #exit()
                # Convert numpy nd_array to torch variables for calculation
                print(f"target_controller_id {target_controller_id} {list(Qs[target_controller_id].parameters())[0]}")
                obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
                act_batch = Variable(torch.from_numpy(act_batch).long())
                rew_batch = Variable(torch.from_numpy(rew_batch))
                #print(rew_batch[0])
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
                qval = Qs[target_controller_id](obs_batch)
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
    
                target_qval = targetQs[target_controller_id](next_obs_batch)
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
                optimizers[target_controller_id].zero_grad()
                # run backward pass
                current_Q_values.backward(d_error.data)
        
                # Perfom the update
                optimizers[target_controller_id].step()
                #print("step")
                num_param_updates_per_controller[target_controller_id] += 1
        
                # Periodically update the target network by Q network to target Q network
                if num_param_updates_per_controller[target_controller_id] % target_update_freq == 0:
                    targetQs[target_controller_id].load_state_dict(Qs[target_controller_id].state_dict())



            target_controller_id = next_target_controller_id[0]        
        
        ##################################################################################################
        #working phase

        env = sim_env(switch_num, controller_num)
        controller_state = env.reset()

        ep_reward = 0
        # pick controller with utilization roulettes
        utilization_per_controller = []
        for k in range(controller_num):
            utilization_per_controller.append(controller_state[k*5+2]+controller_state[k*5+3] + controller_state[k*5+4])
        last_obs = copy.deepcopy(utilization_per_controller)


        replay_buffers_val = []
        for i in range(controller_num):
            # Construct the replay buffer
            replay_buffer = ReplayBuffer(episode_length+1, frame_history_len, input_arg)
            replay_buffers_val.append(replay_buffer)
        for itr in range(episode_length):
            #print(target_controller_id)
            ts[target_controller_id] = ts[target_controller_id]+1
            
            ### Step the env and store the transition
            # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
            # encode_recent_observation will take the latest observation
            # that you pushed into the buffer and compute the corresponding
            # input that should be given to a Q network by appending some
            # previous frames.
    
            # Choose random action if not yet start learning
            last_idx =0
            for i in range(controller_num):
                replay_buffers_val[i].store_frame(last_obs)
                #print(last_obs)
            actions_per_ctrl = []
            for i in range(controller_num):
                recent_observations = replay_buffers_val[i].encode_recent_observation(validation=True)
                #print(i)
                #print(recent_observations)
                
                #print(recent_observations[0])
                #print(recent_observations[1])
                #
                #print(ts[target_controller_id])
                v_actions = 0
                with torch.no_grad():
                    recent_observations = torch.from_numpy(recent_observations).type(dtype).unsqueeze(0)
        
                    recent_observations = recent_observations.view(frame_history_len,  input_arg)
                    recent_observations = recent_observations.unsqueeze(0)
                    
                    #print(f"itr num {itr} : controllerid {i} {recent_observations}")
                    actions = Qs[i](recent_observations)
                    #print(actions)
                    #print(actions)
                    #print(actions)
                    #print(actions)
                    #print(actions)
                    #print(actions)
                    #print(actions.shape)
                    #actions = actions[:,frame_history_len-1,:]
                    v_actions = actions.data.max(1)[1].cpu().unsqueeze(0)
                #v_actions = select_epilson_greedy_action_val(Qs[i], recent_observations, ts[i] - learning_starts)
                #print(actions)
                #print(f"actions {actions} actions.shpae : {actions.shape}")
                v_actions = v_actions[0, 0]
                v_actions = v_actions.item()
                actions_per_ctrl.append(v_actions)
            #print(actions_per_ctrl)
            #print(actions_per_ctrl)
            target_switch_per_action = []
            dest_ctrl_per_action = []
            src_ctrl_per_action = []    
            for i in range(controller_num):

                # if action stay(0) / import(1) /export(2)
                dest_ctrl_id =0
                target_switch_id = 0
                src_ctrl_id = 0
                if(actions_per_ctrl[i] == 0) : 
                    target_switch_id = -1
                    dest_ctrl_id = -1
                    src_ctrl_id = -1
                elif(actions_per_ctrl[i] == 1) :
                    dest_ctrl_id = i
                    nb_per_controller = []
                    cpu_per_controller = []
                    ram_per_controller = []
                    utilization_per_controller = []
                    for k in range(controller_num):
                        nb_per_controller.append(controller_state[k*5+2])
                        cpu_per_controller.append(controller_state[k*5+3])
                        ram_per_controller.append(controller_state[k*5+4])
                        utilization_per_controller.append(controller_state[k*5+2]+controller_state[k*5+3] + controller_state[k*5+4])   
                    src_ctrl_id = utilization_per_controller.index(max(utilization_per_controller))
                    switches_in_src = env.controller_assigned_info[src_ctrl_id]
                    util_gap_for_switches_in_src = []
                    for switch_id in switches_in_src :
                        util_gap = math.pow(nb_per_controller[src_ctrl_id] - nb_per_controller[dest_ctrl_id] - env.nbs[switch_id],2) +math.pow(cpu_per_controller[src_ctrl_id] - cpu_per_controller[dest_ctrl_id] - env.cpus[switch_id],2)  +math.pow(ram_per_controller[src_ctrl_id] - ram_per_controller[dest_ctrl_id] - env.rams[switch_id],2)
                        util_gap_for_switches_in_src.append(util_gap)
                    target_switch_id = switches_in_src[util_gap_for_switches_in_src.index(min(util_gap_for_switches_in_src))]   
                
                else :
                    src_ctrl_id = i
                    nb_per_controller = []
                    cpu_per_controller = []
                    ram_per_controller = []
                    utilization_per_controller = []
                    for k in range(controller_num):
                        nb_per_controller.append(controller_state[k*5+2])
                        cpu_per_controller.append(controller_state[k*5+3])
                        ram_per_controller.append(controller_state[k*5+4])
                        utilization_per_controller.append(controller_state[k*5+2]+controller_state[k*5+3] + controller_state[k*5+4])
                    #print(utilization_per_controller)   
                    dest_ctrl_id = utilization_per_controller.index(min(utilization_per_controller))
                    #print(dest_ctrl_id)
                    switches_in_src = env.controller_assigned_info[src_ctrl_id]
                    util_gap_for_switches_in_src = []
                    for switch_id in switches_in_src :
                        util_gap = math.pow(nb_per_controller[src_ctrl_id] - nb_per_controller[dest_ctrl_id] - env.nbs[switch_id],2) +math.pow(cpu_per_controller[src_ctrl_id] - cpu_per_controller[dest_ctrl_id] - env.cpus[switch_id],2)  +math.pow(ram_per_controller[src_ctrl_id] - ram_per_controller[dest_ctrl_id] - env.rams[switch_id],2)
                        util_gap_for_switches_in_src.append(util_gap)
                    target_switch_id = switches_in_src[util_gap_for_switches_in_src.index(min(util_gap_for_switches_in_src))]    
                target_switch_per_action.append(target_switch_id)
                dest_ctrl_per_action.append(dest_ctrl_id)
                src_ctrl_per_action.append(src_ctrl_id)
            #print(src_ctrl_per_action)
            
            
            #get before usage
            before_nbs = copy.deepcopy(env.nbs)
            before_rams = copy.deepcopy(env.rams)  
            before_cpus = copy.deepcopy(env.cpus)
            before_controller_assigned_info = copy.deepcopy(env.controller_assigned_info)

            before_nb_per_controller = []
            before_cpu_per_controller = []
            before_ram_per_controller = []
            for k in range(controller_num):
                before_nb_per_controller.append(controller_state[k*5+2])
                before_cpu_per_controller.append(controller_state[k*5+3])
                before_ram_per_controller.append(controller_state[k*5+4]) 
            
            new_controller_state = env.update_state_info()

            after_nbs = copy.deepcopy(env.nbs)
            after_rams = copy.deepcopy(env.rams)  
            after_cpus = copy.deepcopy(env.cpus)
            #print(after_nbs[0])
            after_nb_per_controller = []
            after_cpu_per_controller = []
            after_ram_per_controller = []
            for k in range(controller_num):
                after_nb_per_controller.append(new_controller_state[k*5+2])
                after_cpu_per_controller.append(new_controller_state[k*5+3])
                after_ram_per_controller.append(new_controller_state[k*5+4]) 

            reward_per_ctrl = []
            for i in range(controller_num):
                reward = 0
                if(dest_ctrl_per_action[i] is not -1):
                    switches_in_src = before_controller_assigned_info[src_ctrl_per_action[i]]
                    src_ctrl_id = src_ctrl_per_action[i]
                    dest_ctrl_id = dest_ctrl_per_action[i]
                    target_switch_id = target_switch_per_action[i]

                    before_util_diff =  math.pow(before_nb_per_controller[src_ctrl_id] - before_nb_per_controller[dest_ctrl_id],2)  \
                                +math.pow(before_cpu_per_controller[src_ctrl_id] - before_cpu_per_controller[dest_ctrl_id],2)  \
                                +math.pow(before_ram_per_controller[src_ctrl_id] - before_ram_per_controller[dest_ctrl_id],2)
    
                    after_util_diff =  math.pow(after_nb_per_controller[src_ctrl_id] - after_nb_per_controller[dest_ctrl_id]-env.nbs[target_switch_id],2)  \
                                +math.pow(after_cpu_per_controller[src_ctrl_id] - after_cpu_per_controller[dest_ctrl_id]-env.cpus[target_switch_id],2)  \
                                +math.pow(after_ram_per_controller[src_ctrl_id] - after_ram_per_controller[dest_ctrl_id]-env.rams[target_switch_id],2)   
    
                    reward =  before_util_diff - after_util_diff
                reward_per_ctrl.append(reward)

            selected_action = reward_per_ctrl.index(max(reward_per_ctrl))
            if(dest_ctrl_per_action[selected_action] is not -1):
                #remove switches from 
                env.controller_assigned_info[src_ctrl_per_action[selected_action]].remove(target_switch_per_action[selected_action])
                #add switch to dest ctrl
                env.controller_assigned_info[dest_ctrl_per_action[selected_action]].extend([target_switch_per_action[selected_action]])
                #print(self.controller_assigned_info)
                env.switches_assigned_info[target_switch_per_action[selected_action]] = dest_ctrl_per_action[selected_action]
                #print(env.controller_assigned_info)
            new_controller_state = env.get_state_info()
           
            ep_reward = ep_reward + max(reward_per_ctrl)
            #print(reward_per_ctrl)
            #print(max(reward_per_ctrl))   
            controller_state = new_controller_state
            utilization_per_controller = []
            for k in range(controller_num):
                utilization_per_controller.append(controller_state[k*5+2]+controller_state[k*5+3] + controller_state[k*5+4])
            last_obs = copy.deepcopy(utilization_per_controller) 
            #print(last_obs)  
            #print(last_obs)           
            if(episode == 0):
                ctrls = []
                for ctrl_id in env.controller_assigned_info :
                    switches = env.controller_assigned_info[ctrl_id]
                    ctrl_util = 0
                    ctrl_ram = 0
                    ctrl_cpu = 0
                    ctrl_nb = 0 
                    for i in switches :
                        ctrl_cpu = ctrl_cpu + env.cpus[i]
                        ctrl_ram = ctrl_ram + env.rams[i]
                        ctrl_nb = ctrl_nb + env.nbs[i]
                    ctrl_util = ctrl_cpu + ctrl_ram + ctrl_nb
                    ctrls.extend([ctrl_cpu, ctrl_ram, ctrl_nb, ctrl_util])
                with open('after_marvel.csv', '+a', newline='') as csvfile:    
                    workloadwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    workloadwriter.writerow(ctrls)        
        with open(f'/scratch/x2026a02/marvel_controller_loadbalnacer/_{switch_num}_{controller_num}_marvel4.csv', '+a') as f:
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
            for i in range(controller_num):
                # Dump statistics to pickle
                torch.save(Qs[i].state_dict(), f"/scratch/x2026a02/marvel_controller_loadbalnacer/{episode}_{switch_num}_{controller_num}_Q{i}.pth")
                torch.save(targetQs[i].state_dict(), f"/scratch/x2026a02/marvel_controller_loadbalnacer/{episode}_{switch_num}_{controller_num}_target_Q{i}.pth")
    for i in range(controller_num):            
        # Dump statistics to pickle
        torch.save(Qs[i].state_dict(), f"/scratch/x2026a02/marvel_controller_loadbalnacer/{episode}_{switch_num}_{controller_num}_Q{i}.pth")
        torch.save(targetQs[i].state_dict(), f"/scratch/x2026a02/marvel_controller_loadbalnacer/{episode}_{switch_num}_{controller_num}_target_Q{i}.pth")
    