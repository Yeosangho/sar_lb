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
import csv
from sim_env import sim_env
import subprocess
from subprocess import Popen, PIPE
import random
import time
from torch.nn.utils.rnn import pack_sequence
import copy
import math
from pandas import Series, DataFrame
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

#def make_observations(recent_observations,  i, last_obs):
    
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
    for out_out_iter in range(2):
        times = 1
        highscore = 15
        if(out_out_iter == 0):
            burst = False
        else :
            burst = True 
        for out_iter in range(4):
            
            for in_iter in range(4):
                rseed = in_iter + 777
                env_switch_num = 20 * times
                num_actions = controller_num  = 4 * times# 4 controllers
                input_arg = (controller_num+1)*5 #(4+1)*5
                switch_num = env_switch_num
                # Construct an epilson greedy policy with given exploration schedule
                def select_epilson_greedy_action(model, obs, t):
                    sample = random.random()
                    #print(f"obs.shape {obs.shape}")
                    obs = torch.from_numpy(obs).type(dtype)
                    #print(obs.shape)
                    # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
                    with torch.no_grad():
                        hidden = torch.zeros(obs.shape, requires_grad = True)
                        obs = obs.view(int(obs.shape[0]),frame_history_len,  input_arg)
                        #obs = obs.unsqueeze(0)
                        actions = model( Variable(obs) )
                        #print(actions.shape)
                        #actions = actions[:,frame_history_len-1,:]
                        return actions.data.max(1)[1].cpu().unsqueeze(0)
            
            
                # Initialize target q function and q function
                Q = q_func(input_arg, num_actions).type(dtype)
                target_Q = q_func(input_arg, num_actions).type(dtype)
                
                if out_iter > 0 :
                    Q.load_state_dict(torch.load(f"9000_{switch_num}_{controller_num}_Q.pth"))
                else : 
                    Q.load_state_dict(torch.load(f"sto_9999_Q.pth"))
                Q.eval()
                if out_iter > 0 :
                    target_Q.load_state_dict(torch.load(f"9000_{switch_num}_{controller_num}_target_Q.pth"))
                else : 
                    target_Q.load_state_dict(torch.load(f"sto_9999_target_Q.pth"))
                target_Q.eval()    
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
                
                for episode in range(1):
                    env = sim_env(env_switch_num, controller_num, rseed, burst)
                    last_obs = env.reset()
                    ep_reward = 0
                    temp_last_obs = copy.deepcopy(last_obs)
                    temp_last_obs["controllers"].extend(temp_last_obs["switches"][0])
                    idx = env.switches_assigned_info[0]
                    temp_last_obs["controllers"][idx*5+2] = temp_last_obs["controllers"][idx*5+2] -temp_last_obs["switches"][0][2]
                    temp_last_obs["controllers"][idx*5+3] = temp_last_obs["controllers"][idx*5+3] -temp_last_obs["switches"][0][3]
                    temp_last_obs["controllers"][idx*5+4] = temp_last_obs["controllers"][idx*5+4] -temp_last_obs["switches"][0][4]
                #
                    
                    temp_last_obs = temp_last_obs["controllers"]    
                    replay_buffer.store_frame(temp_last_obs)        
                    #switch_id = env.utilization.index(max(env.utilization))
            
                    #print(last_obs[0])
            
                    start_time = time.time()
                    rows = [] 
                    for itr in range(episode_length):
                        t = t+1
                        ### Step the env and store the transition
                        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
                        # encode_recent_observation will take the latest observation
                        # that you pushed into the buffer and compute the corresponding
                        # input that should be given to a Q network by appending some
                        # previous frames.
                        actions_per_switch = []
                        random.seed(t)
            
                        utilization_per_controller = []
                        for k in range(controller_num):
                            utilization_per_controller.append(last_obs["controllers"][k*5+2] + last_obs["controllers"][k*5+3] + last_obs["controllers"][k*5+4])
                        #select most overloaded controller
                        switches = []
                        for i in range(int(controller_num * 1)):
                            ctrl_id = utilization_per_controller.index(max(utilization_per_controller))
                            utilization_per_controller.remove(max(utilization_per_controller))
                            switches.extend(env.controller_assigned_info[ctrl_id])
                        #print(switches)
                        # Choose random action if not yet start learning
                        obss_switchs = None
                        recent_observations  = replay_buffer.encode_recent_observation(validation=True).tolist()
                        #print(np.array(recent_observations).shape)
                        temp_last_obs = copy.deepcopy(last_obs)
                        temp_last_obs["controllers"].extend([0,0,0,0,0])
                        #print(np.array(temp_last_obs["controllers"]).shape)
                        #print(np.array([recent_observations]).shape)
                        recent_observations[frame_history_len-1] =  temp_last_obs["controllers"]
                        recent_observations =  np.array([recent_observations] * len(switches))
                        #print(recent_observations[2][frame_history_len-1] )
                        #print(np.array(recent_observations).shape)
                        #temp_last_obs = copy.deepcopy(last_obs)
                        #temp_last_obs["controllers"].extend(temp_last_obs["switches"][0])
                        #recent_observations[frame_history_len-1] = temp_last_obs["controllers"]
                        
                        
                        #obss_switchs = np.array(recent_observations)
                        #print(obss_switchs.shape)
                        #recent_observations = recent_observations[1:frame_history_len]
                        swithes_count =0 
                        #recent_observations = [make_observations(recent_observations, i, last_obs) for i in switches ]
                        #temp_last_obs["controllers"].extend(temp_last_obs["switches"][i])  
                        switches_info = np.array([temp_last_obs["switches"][i] for i in switches])
                        #print(switches_info[2])
                        #print(np.array(switches_info).shape)
                        stacked_last_obs = []    
                        temp_last_obs = np.array(temp_last_obs["controllers"])
                        for info, i in zip(switches_info, switches) :
                            local_last_obs = copy.deepcopy(temp_last_obs)
                            idx = env.switches_assigned_info[i]
                            #print(info.shape)
                            local_last_obs[controller_num*5:controller_num*5+5] += info
                            local_last_obs[idx*5+2:idx*5+5] -= info[2:5]
                            stacked_last_obs.append(local_last_obs.tolist())

                        #if(itr == 2):
                        #    break
                        #reduce(lambda x, y: [y] if not x else x + [y + x[-1]], temp_last_obs, None)              
                        #temp_last_obs = []
                        #print(np.array(temp_last_obs).shape)
                            #idx = env.switches_assigned_info[i]
                            #print(last_obs["controllers"])
                            #print(last_obs["switches"][i])
                            
            ###             
                            #temp_last_obs = np.array(temp_last_obs["controllers"])
                           
                        #print(np.array(temp_last_obs))
                        #print(temp_last_obs.shape)
                       #print(recent_observations[:,frame_history_len-1].shape)    
                        recent_observations[:,frame_history_len-1]  = np.array(stacked_last_obs)
       ##               recent_observations[]
                            #last_idx = replay_buffer.store_frame(temp_last_obs)
#
                            #recent_observations[swithes_count][frame_history_len-1] = temp_last_obs
                            #swithes_count = swithes_count + 1
                            #replay_buffer.go_back()
                        #print(obss_switchs.shape)
                        #print(recent_observations[2][frame_history_len-1])
                        obss_switchs = np.array(recent_observations, dtype=np.float32)
                        

                        actions = select_epilson_greedy_action(Q, obss_switchs, t- learning_starts)
                        #print(f"actions {actions} actions.shpae : {actions.shape}")
                        action = actions[0]
                        #print(action)
                        actions_per_switch.extend(action)                        
                        #get before usage
                        controller_state = env.get_state_info()
                        before_nbs = copy.deepcopy(env.nbs)
                        before_rams = copy.deepcopy(env.rams)  
                        before_cpus = copy.deepcopy(env.cpus)
                        before_controller_assigned_info = copy.deepcopy(env.controller_assigned_info)
            #
                        before_nb_per_controller = []
                        before_cpu_per_controller = []
                        before_ram_per_controller = []
                        for k in range(controller_num):
                            before_nb_per_controller.append(controller_state[k*5+2])
                            before_cpu_per_controller.append(controller_state[k*5+3])
                            before_ram_per_controller.append(controller_state[k*5+4]) 
                        
                        new_controller_state = env.update_state_info()
            #
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
            #
                        reward_per_switch = []
                        src_ctrl_per_switch = []
                        dest_ctrl_per_switch = []
            #
            #
                        
                        for i in range(len(switches)) :
                            reward = 0
                #
                            switch_id = switches[i] 
                            dest_ctrl_id = actions_per_switch[i].item()
                            src_ctrl_id = env.switches_assigned_info[switch_id]
            #
                            before_util_diff =  math.pow(before_nb_per_controller[src_ctrl_id] - before_nb_per_controller[dest_ctrl_id],2)  \
                                        +math.pow(before_cpu_per_controller[src_ctrl_id] - before_cpu_per_controller[dest_ctrl_id],2)  \
                                        +math.pow(before_ram_per_controller[src_ctrl_id] - before_ram_per_controller[dest_ctrl_id],2)
                #
                            after_util_diff =  math.pow(after_nb_per_controller[src_ctrl_id] - after_nb_per_controller[dest_ctrl_id]-env.nbs[switch_id],2)  \
                                        +math.pow(after_cpu_per_controller[src_ctrl_id] - after_cpu_per_controller[dest_ctrl_id]-env.cpus[switch_id],2)  \
                                        +math.pow(after_ram_per_controller[src_ctrl_id] - after_ram_per_controller[dest_ctrl_id]-env.rams[switch_id],2)   
                #
                            reward =  before_util_diff - after_util_diff
            #
                            reward_per_switch.append(reward)
                            src_ctrl_per_switch.append(src_ctrl_id)
                            dest_ctrl_per_switch.append(dest_ctrl_id)
            #
                        selected_idx = reward_per_switch.index(max(reward_per_switch))
                        selected_switch = switches[reward_per_switch.index(max(reward_per_switch))]
                        #print(selected_switch)
            
                        selected_ctrl = env.switches_assigned_info[selected_switch]
                        last_obs["controllers"].extend(last_obs["switches"][selected_switch])
                        last_obs["controllers"][selected_ctrl*5+2] = last_obs["controllers"][selected_ctrl*5+2] -last_obs["switches"][selected_switch][2]
                        last_obs["controllers"][selected_ctrl*5+3] = last_obs["controllers"][selected_ctrl*5+3] -last_obs["switches"][selected_switch][3]
                        last_obs["controllers"][selected_ctrl*5+4] = last_obs["controllers"][selected_ctrl*5+4] -last_obs["switches"][selected_switch][4]
                #
                        
                        last_obs = last_obs["controllers"]
            #
                        last_idx = replay_buffer.store_frame(last_obs)
            #
            #
                        #remove switches from 
                        env.controller_assigned_info[src_ctrl_per_switch[selected_idx]].remove(selected_switch)
                        #add switch to dest ctrl
                        #print(dest_ctrl_per_switch[selected_idx])
                        env.controller_assigned_info[dest_ctrl_per_switch[selected_idx]].append(selected_switch)
                        #print(self.controller_assigned_info)
                        env.switches_assigned_info[selected_switch] = dest_ctrl_per_switch[selected_idx]
                        #print(env.controller_assigned_info)
            #
                        new_controller_state = env.get_state_info()
                        last_obs = env.get_obs()
                        controller_state = new_controller_state
                        
                        #print(obs)
                        if(itr == (episode_length-1)):
                            done = 1
                        else :
                            done = 0
                        # clip rewards between -1 and 1
                        reward = max(reward_per_switch)
                        ep_reward = ep_reward + reward
                        reward = reward *10
                        reward = max(-1.0, min(reward, 1.0))
                        # Store other info in replay memory
                        replay_buffer.store_effect(last_idx, dest_ctrl_per_switch[selected_idx], reward, done)
                        # Resets the environment when reaching an episode boundary.
            
                        ### Perform experience replay and train the network.
                        # Note that this is only done if the replay buffer contains enough samples
                        # for us to learn something useful -- until then, the model will not be
                        # initialized and random actions should be taken
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
                    with open(f'after_burst_{burst}_{env_switch_num}_{controller_num}_{rseed}_rl3_ep_info.csv', 'w+', newline='') as f: 
                        print(f"{ep_reward},{time.time() - start_time}", file=f)   
    
                    print(f"episode time : {time.time() - start_time}")        
                    print(ep_reward)
    
                    with open(f'after_burst_{burst}_{env_switch_num}_{controller_num}_{rseed}_rl3.csv', 'w+', newline='') as csvfile:    
                        workloadwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        for i in range(300):
                            workloadwriter.writerow(rows[i])
            
    
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
            times = 2 * times    
    