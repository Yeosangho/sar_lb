import argparse

from sim_env import sim_env
import random 
import csv
import time
import copy
import math

def main():

	for l2_iter in range(2):
	
		burst = False
		if(l2_iter == 0):
			burst = False
		else :
			burst = True
		
		times = 1
		rseed = 777	
		for x in range(4):
			t =0
			controller_num = num_actions = 4 * times
			env_switch_num = 20 * times
			switch_num =env_switch_num
	
			for i in range(4):
				rseed = 777 + i
				env = sim_env(switch_num, controller_num, rseed, burst)
				last_obs = env.reset()
				ep_reward = 0
				start_time = time.time()
				rows = []
				for j in range(300):
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
					ctrl_id = utilization_per_controller.index(max(utilization_per_controller))
					switches = env.switches  
	
					
					#get before usage
					controller_state = env.get_state_info()
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
					
					reward_per_switch = []
					src_ctrl_per_switch = []
					dest_ctrl_per_switch = []
					
					
					
					for i in range(len(switches)) :
						reward = 0
					
						switch_id = i
						src_ctrl_id = env.switches_assigned_info[switch_id]
						max_reward = -10000
						max_reward_dest_id = -1
						for j in range(controller_num):
							dest_ctrl_id = j
							before_util_diff =  math.pow(before_nb_per_controller[src_ctrl_id] - before_nb_per_controller[dest_ctrl_id],2)  \
							            +math.pow(before_cpu_per_controller[src_ctrl_id] - before_cpu_per_controller[dest_ctrl_id],2)  \
							            +math.pow(before_ram_per_controller[src_ctrl_id] - before_ram_per_controller[dest_ctrl_id],2)
					
							after_util_diff =  math.pow(after_nb_per_controller[src_ctrl_id] - after_nb_per_controller[dest_ctrl_id]-env.nbs[switch_id],2)  \
							            +math.pow(after_cpu_per_controller[src_ctrl_id] - after_cpu_per_controller[dest_ctrl_id]-env.cpus[switch_id],2)  \
							            +math.pow(after_ram_per_controller[src_ctrl_id] - after_ram_per_controller[dest_ctrl_id]-env.rams[switch_id],2)   
					
							reward =  before_util_diff - after_util_diff
							if(reward > max_reward):
								max_reward = reward
								max_reward_dest_id = dest_ctrl_id 
							
						reward_per_switch.append(max_reward)
						src_ctrl_per_switch.append(src_ctrl_id)
						dest_ctrl_per_switch.append(max_reward_dest_id)
					
					selected_idx = reward_per_switch.index(max(reward_per_switch))
					selected_switch = reward_per_switch.index(max(reward_per_switch))
					
					
					selected_ctrl = env.switches_assigned_info[selected_switch]
					last_obs["controllers"].extend(last_obs["switches"][selected_switch])
					last_obs["controllers"][selected_ctrl*5+2] = last_obs["controllers"][selected_ctrl*5+2] -last_obs["switches"][selected_switch][2]
					last_obs["controllers"][selected_ctrl*5+3] = last_obs["controllers"][selected_ctrl*5+3] -last_obs["switches"][selected_switch][3]
					last_obs["controllers"][selected_ctrl*5+4] = last_obs["controllers"][selected_ctrl*5+4] -last_obs["switches"][selected_switch][4]
					
					
					last_obs = last_obs["controllers"]
					
					
					
					#remove switches from 
					env.controller_assigned_info[src_ctrl_per_switch[selected_idx]].remove(selected_switch)
					#add switch to dest ctrl
					env.controller_assigned_info[dest_ctrl_per_switch[selected_idx]].append(selected_switch)
					#print(self.controller_assigned_info)
					env.switches_assigned_info[selected_switch] = dest_ctrl_per_switch[selected_idx]
					#print(env.controller_assigned_info)
					
					new_controller_state = env.get_state_info()
					last_obs = env.get_obs()
					controller_state = new_controller_state
					
					#print(obs)
		
					# clip rewards between -1 and 1
					reward = max(reward_per_switch)
					ep_reward = ep_reward + reward
					reward = reward *10
					reward = max(-1.0, min(reward, 1.0))
					# Store other info in replay memory
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
				with open(f'after_burst_{burst}_{env_switch_num}_{controller_num}_{rseed}_elastic_ep_info.csv', 'w', newline='') as f:  
					print(f'{ep_reward}, {time.time()-start_time}', file=f)  
					
					
				with open(f'after_burst_{burst}_{env_switch_num}_{controller_num}_{rseed}_elastic.csv', 'w', newline='') as csvfile:    
					workloadwriter = csv.writer(csvfile, delimiter=',',
						            quotechar='|', quoting=csv.QUOTE_MINIMAL)
					for i in range(300):
						workloadwriter.writerow(rows[i])
 	



			times = times * 2

if __name__ == '__main__':
	main()