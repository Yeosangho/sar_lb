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
import math
import copy
import csv
import torch.optim as optim

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

    for out_out_iter in range(2):
        times = 1
        highscore = 15
        burst = True
        if(out_out_iter == 0):
            burst = False
    
        for out_iter in range(4):
            
            for in_iter in range(4):
                rseed = in_iter + 777
                env_switch_num = 20 * times
                num_actions =3
                controller_num  = 4 * 1 # 4 controllers
                input_arg = controller_num
                switch_num = env_switch_num
    
                # Construct an epilson greedy policy with given exploration schedule
                def select_epilson_greedy_action(model, obs, t, target_controller_id):
                    sample = random.random()
                    eps_threshold = exploration.value(t)
                    if sample > eps_threshold:
                        #print(f"obs.shape {obs.shape}")
                        obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)
                        # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
                        with torch.no_grad():
                            obs = obs.view(frame_history_len,  input_arg)
                            obs = obs.unsqueeze(0)
                            actions = model( Variable(obs))
                            #print(f"cid {target_controller_id} {actions}")
                            #print(actions.shape)
                            #actions = actions[:,frame_history_len-1,:]
                            return actions.data.max(1)[1].cpu().unsqueeze(0)
                    else:
                        return torch.IntTensor([[random.randrange(num_actions)]])
            
                # Initialize target q function and q function
                Q = []
                target_Q = []
                optimizer = []
                replay_buffer = []
                num_param_updates = [0] * controller_num
                t = [0] * controller_num
                for i in range(controller_num) :
                    q = q_func(input_arg, num_actions).type(dtype)
                    if out_iter > 0 :
                        q.load_state_dict(torch.load(f"1000_{switch_num}_{controller_num}_Q{i}.pth"))
                    else :
                        q.load_state_dict(torch.load(f"4000_{switch_num}_{controller_num}_Q{i}.pth"))
                    q.eval()        
                    q = copy.deepcopy(q)
                    tq = q_func(input_arg, num_actions).type(dtype)
                    if out_iter > 0 :
                        tq.load_state_dict(torch.load(f"1000_{switch_num}_{controller_num}_target_Q{i}.pth"))
                    else : 
                        tq.load_state_dict(torch.load(f"4000_{switch_num}_{controller_num}_target_Q{i}.pth"))
                    tq.eval()
            
                    # Construct Q network optimizer function
                    optimizer_ctrl = optim.RMSprop(params=q.parameters(), lr=0.01, alpha=0.95, eps= 0.01)
                
                    # Construct the replay buffer
                    replay_buffer_ctrl = ReplayBuffer(replay_buffer_size, frame_history_len, input_arg)
                    
                    Q.append(q)
                    target_Q.append(tq)
                    optimizer.append(optimizer_ctrl)
                    replay_buffer.append(replay_buffer_ctrl) 
                ###############
                # RUN ENV     #
                ###############
                mean_episode_reward = -float('nan')
                best_mean_episode_reward = -float('inf')
                LOG_EVERY_N_STEPS = 10000
                episode_length = 300
                start_time = 0
                for episode in range(1):
                    env = sim_env(env_switch_num, controller_num, rseed, burst)
                    controller_state = env.reset()
                    ep_reward = 0
            
            
                    ##################################################################################################
                    #working phase
                    start_time = time.time()
                    ep_reward = 0
                    # pick controller with utilization roulettes
                    utilization_per_controller = []
                    for k in range(controller_num):
                        utilization_per_controller.append(controller_state[k*5+2]+controller_state[k*5+3] + controller_state[k*5+4])
                    last_obs = copy.deepcopy(utilization_per_controller)
            
            
                    replay_buffer_val = []
                    for i in range(controller_num):
                        # Construct the replay buffer
                        rb = ReplayBuffer(episode_length+1, frame_history_len, input_arg)
                        replay_buffer_val.append(rb)
                    rows = []
                    for itr in range(episode_length):
                        #print(target_controller_id)
            
                        ### Step the env and store the transition
                        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
                        # encode_recent_observation will take the latest observation
                        # that you pushed into the buffer and compute the corresponding
                        # input that should be given to a Q network by appending some
                        # previous frames.
                
                        # Choose random action if not yet start learning
                        last_idx =0
                        for i in range(controller_num):
                            replay_buffer_val[i].store_frame(last_obs)
                            #print(last_obs)
                        actions_per_ctrl = []
                        for i in range(controller_num):
                            recent_observations = replay_buffer_val[i].encode_recent_observation(validation=True)
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
                                
                                actions = Q[i](copy.deepcopy(recent_observations))
                                #actions = actions[:,frame_history_len-1,:]
                                actions = actions.data.max(1)[1].cpu().unsqueeze(0)
                            #v_actions = select_epilson_greedy_action_val(Qs[i], recent_observations, ts[i] - learning_starts)
                            #print(f"actions {actions} actions.shpae : {actions.shape}")
                            actions = actions[0, 0]
                            actions = actions.item()
                            actions_per_ctrl.append(actions)
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
                                dest_ctrl_id = utilization_per_controller.index(min(utilization_per_controller))
                                switches_in_src = env.controller_assigned_info[src_ctrl_id]
                                util_gap_for_switches_in_src = []
                                for switch_id in switches_in_src :
                                    util_gap = math.pow(nb_per_controller[src_ctrl_id] - nb_per_controller[dest_ctrl_id] - env.nbs[switch_id],2) +math.pow(cpu_per_controller[src_ctrl_id] - cpu_per_controller[dest_ctrl_id] - env.cpus[switch_id],2)  +math.pow(ram_per_controller[src_ctrl_id] - ram_per_controller[dest_ctrl_id] - env.rams[switch_id],2)
                                    util_gap_for_switches_in_src.append(util_gap)
                                target_switch_id = switches_in_src[util_gap_for_switches_in_src.index(min(util_gap_for_switches_in_src))]    
                            target_switch_per_action.append(target_switch_id)
                            dest_ctrl_per_action.append(dest_ctrl_id)
                            src_ctrl_per_action.append(src_ctrl_id)
                        
                        
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
                            rows.append(ctrls)    
                    with open(f'after_burst_{burst}_{env_switch_num}_{controller_num}_{rseed}_marvel.csv', 'w+', newline='') as csvfile:    
                        workloadwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        for i in range(300):
                            workloadwriter.writerow(rows[i])        
            
                    ### 4. Log progress and keep track of statistics
                    episode_rewards = ep_reward
                    with open(f'/home/soboru963/marvel_controller_loadbalnacer_val/after_burst_{burst}_{env_switch_num}_{controller_num}_{rseed}_marvel_ep_info.csv', 'w+') as f:
                        print('{score}, {ep_time}'
                            .format(ep=episode,score=ep_reward, ep_time=time.time()-start_time),
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
            times = times * 2 



