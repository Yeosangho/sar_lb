from sim_env import sim_env
import csv
import random 
import time
def main():
	t =0
	for out_iter in range(2):
		times = 1
		burst = False
		if(out_iter == 0):
			burst = False
		else :
			burst = True

		for x in range(4):
			
			controller_num = num_actions = 4 * times
			switch_num = 20 * times
			rseed_start = 777
			for i in range(4):
				rseed = rseed_start + i
				env = sim_env(switch_num, controller_num, rseed, burst)
				state = env.reset()
				ep_reward = 0
				start_time = time.time()
				rows = []
				for j in range(300):
					t = t +1 
					random.seed(t)
					
					controller_index = list(range(controller_num))	
		
					controller_assigned_info = env.controller_assigned_info
					utilization_per_controller = []
					if(j >= 0):
						for k in range(controller_num):
							utilization_per_controller.append(state["controllers"][k*5+2]+state["controllers"][k*5+3] + state["controllers"][k*5+4])
					else :
						for k in range(controller_num):
							utilization_per_controller.append(state[k*5+2]+state[k*5+3] + state[k*5+4])
		
					#select most overloaded controller
					idx = utilization_per_controller.index(max(utilization_per_controller))
					
		
					#select switch in overloaded controller 
					utilization_per_switch_in_controller = []
		
					switches = controller_assigned_info[idx]
					for switch_idx in switches :
						utilization_per_switch_in_controller.append(env.nbs[switch_idx] + env.cpus[switch_idx] +env.rams[switch_idx])
		
						
					target_switch  = switches[utilization_per_switch_in_controller.index(max(utilization_per_switch_in_controller))]	
					#print(target_switch)
					#find_	source ctrl
					src_ctrl = env.switches_assigned_info[target_switch]
					#set target ctrl
					target_ctrl = (src_ctrl + 1) % controller_num
					_, state, reward = env.step(target_switch, target_ctrl)
		
					ep_reward = ep_reward + reward
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
				with open(f'after_burst_{burst}_{switch_num}_{controller_num}_{rseed}_rr_ep_info.csv', 'w+', newline='') as f :
					print(f"{ep_reward},{time.time() - start_time}", file=f)        
				with open(f'after_burst_{burst}_{switch_num}_{controller_num}_{rseed}_rr.csv', 'w+', newline='') as csvfile:    
				    workloadwriter = csv.writer(csvfile, delimiter=',',
				                quotechar='|', quoting=csv.QUOTE_MINIMAL)
				    for i in range(300):
				    	workloadwriter.writerow(rows[i])
			times = times * 2

if __name__ == '__main__':
	main()