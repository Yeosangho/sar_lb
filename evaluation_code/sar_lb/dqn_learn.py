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
                num_actions = controller_num  = 4 * 1 # 4 controllers
                input_arg = (controller_num+1)*5 #(4+1)*5
                switch_num = env_switch_num
                # Construct an epilson greedy policy with given exploration schedule
                def select_epilson_greedy_action(model, obs, t):
                    sample = random.random()
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
            
            
                # Initialize target q function and q function
                Q = q_func(input_arg, num_actions).type(dtype)
                target_Q = q_func(input_arg, num_actions).type(dtype)
                
                if out_iter > 0 :
                    Q.load_state_dict(torch.load(f"1000_{switch_num}_{controller_num}_Q.pth"))
                else : 
                    Q.load_state_dict(torch.load(f"sto_9999_Q.pth"))
                Q.eval()
                if out_iter > 0 :
                    target_Q.load_state_dict(torch.load(f"1000_{switch_num}_{controller_num}_target_Q.pth"))
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
            
                    #switch_id = env.utilization.index(max(env.utilization))
            
                    #print(last_obs[0])
                    utilization_per_controller = []
                    for k in range(controller_num):
                        utilization_per_controller.append(last_obs["controllers"][k*5+2]+last_obs["controllers"][k*5+3] + last_obs["controllers"][k*5+4])
                    #select most overloaded controller
                    idx = utilization_per_controller.index(max(utilization_per_controller))
                    #get switch list from src ctrl
                    switches = env.controller_assigned_info[idx]
                    switch_id = random.choice(switches)


                    last_obs["controllers"].extend(last_obs["switches"][switch_id])
                    last_obs["controllers"][idx*5+2] = last_obs["controllers"][idx*5+2] -last_obs["switches"][switch_id][2]
                    last_obs["controllers"][idx*5+3] = last_obs["controllers"][idx*5+3] -last_obs["switches"][switch_id][3]
                    last_obs["controllers"][idx*5+4] = last_obs["controllers"][idx*5+4] -last_obs["switches"][switch_id][4]
            
                    
                    last_obs = last_obs["controllers"]
                       




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
                        last_idx = replay_buffer.store_frame(last_obs)
                        recent_observations = replay_buffer.encode_recent_observation()
                        random.seed(t)
            
                        #print(recent_observations[0])
                        #print(recent_observations[1])
                        actions = select_epilson_greedy_action(Q, recent_observations, t- learning_starts)
                        #print(f"actions {actions} actions.shpae : {actions.shape}")
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
                    with open(f'after_burst_{burst}_{env_switch_num}_{controller_num}_{rseed}_rl3_randomselect_ep_info.csv', 'w+', newline='') as f: 
                        print(f"{ep_reward},{time.time() - start_time}", file=f)   
    
                    print(f"episode time : {time.time() - start_time}")        
                    print(ep_reward)
    
                    with open(f'after_burst_{burst}_{env_switch_num}_{controller_num}_{rseed}_rl3_randomselect.csv', 'w+', newline='') as csvfile:    
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
    