from sim_env import sim_env

import random 
def main():
	t =0
	times = 1
	for x in range(5):
		times = times * 2
		controller_num = num_actions = 4 * times
		switch_num = 20 * times
		for i in range(10000):
			env = sim_env(switch_num, controller_num)
			state = env.reset()
			ep_reward = 0
			for j in range(300):
				t = t +1 
				random.seed(t)
				
				controller_index = list(range(controller_num))	
	
				controller_assigned_info = env.controller_assigned_info
				utilization_per_controller = []
				if(j > 0):
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
	
			with open(f'/scratch/x2026a02/stochastic_controller_loadbalnacer/_{controller_num}_{switch_num}_roundrobin.csv', '+a') as f:
				print('{ep}, {score}'.format(ep=i,score=ep_reward), file=f )

if __name__ == '__main__':
	main()