import csv
import numpy as np 
burst = False
rseed_start = 777
random_seed_num = 4
prefix = "after_"
marvel_post_fix = "_marvel"
rl_post_fix = "_rl3"
rl_random_post_fix = "_rl3_randomselect"
rr_post_fix = "_rr"
dha_post_fix = "_DHA_LB"
post_fixs = ["_marvel", "_rl3", "_rl3_randomselect", "_rr", "_DHA_LB", "_elastic", "_noloadbalance"]
folder_first_names = ["marvel", "1221_rl", "rl_randomselect", "rr", "dha", "elastic", "nobalance"]
folder_second_names = ["switch", "switch_ctrl"]
folder_last_name = "increasement"


simtime_folder_name = "simulation_time_without_policy"

ep_info = "ep_info"


for folder_second_name in folder_second_names :
	if(folder_second_name == "switch"):
	
	
		for out_iter in range(2):
	
			if(out_iter == 0):
				burst = False
			else :
				burst = True
			controller_num = 4
			switch_num = 20	
	
			list_sto_avgs = []
			list_sto_times = []
			for i in range(4):
				stochastic_avgs = []
				stochastic_avgs_10steps = []
				stochastic_times = []
				stochastic_scores = []
				for folder_first_name, post_fix in zip(folder_first_names, post_fixs) : 
					folder_name = folder_first_name + "_" + folder_second_name + "_" + folder_last_name
					stochastic_avgs_10step = []

					avgs = [] 
					times = []
					scores = []
					stddevs = []
					for j in range(random_seed_num):
						loads = [ [0] * controller_num for _ in range(300)]
						rseed = rseed_start + j
						#print(f"{folder_name}/{prefix}burst_{burst}_{switch_num}_{controller_num}_{rseed}{post_fix}.csv")
						file1 = open(f"{folder_name}/{prefix}burst_{burst}_{switch_num}_{controller_num}_{rseed}{post_fix}.csv", 'r') 
						for step in range(300):
							lines = file1.readline().split(",")
							for k in range(controller_num):
								loads[step][k] = float(lines[4*k +3])
						file2 = open(f"{folder_name}/{prefix}burst_{burst}_{switch_num}_{controller_num}_{rseed}{post_fix}_{ep_info}.csv", 'r')
						score = ""
						time = ""
						if(folder_first_name == "nobalance"):
							time =  file2.readline()
						else :
							score_and_time = file2.readline().split(",")
							score = score_and_time[0]
							time = score_and_time[1]
						loads = np.array(loads)
						stddev = np.std(loads, axis=1, ddof=1)
						stddevs.append(stddev)
						avg = np.average(stddev)
						avgs.append(avg)
						times.append(float(time))
						#scores.append(float(score))
					a = np.average(np.array(stddevs), axis=0)
					steps = 0
					for i in range(300-steps):
						if(steps > 0):
							stochastic_avgs_10step.append(np.average(np.array(a[i:i+steps])))
						else :
							stochastic_avgs_10step.append(np.array(a[i]))
					stochastic_avg = sum(avgs) / len(avgs)
					stochastic_avgs.append(stochastic_avg)
					stochastic_times.append(sum(times) / len(times))
					stochastic_avgs_10steps.append(stochastic_avgs_10step)
				print(np.array(stochastic_avgs_10steps).shape)
				with open(f"1221_burst_{burst}_{switch_num}_{controller_num}_{steps}.csv", 'w+', newline='') as csvfile:
					workloadwriter = csv.writer(csvfile, delimiter=',',
					            quotechar='|', quoting=csv.QUOTE_MINIMAL)
					for i in stochastic_avgs_10steps :
						workloadwriter.writerow(i)

	
				list_sto_avgs.append(stochastic_avgs)
				list_sto_times.append(stochastic_times)


				switch_num = switch_num * 2				
			with open(f"1221_burst_{burst}_{folder_second_name}_avg.csv", 'w+', newline='') as csvfile:
				workloadwriter = csv.writer(csvfile, delimiter=',',
				            quotechar='|', quoting=csv.QUOTE_MINIMAL)
				for sto_avgs in list_sto_avgs  :
					workloadwriter.writerow(sto_avgs)
			with open(f"1221_burst_{burst}_{folder_second_name}_time.csv", 'w+', newline='') as csvfile:
				workloadwriter = csv.writer(csvfile, delimiter=',',
				            quotechar='|', quoting=csv.QUOTE_MINIMAL)
				for sto_times in list_sto_times  :
					workloadwriter.writerow(sto_times)


	else :
	
		for out_iter in range(2):
	
			if(out_iter == 0):
				burst = False
			else :
				burst = True
			controller_num = 4
			switch_num = 20	
	
			list_sto_avgs = []
			list_sto_times = []
			for i in range(4):
				stochastic_avgs = []
				stochastic_avgs_10steps = []
				stochastic_times = []
				stochastic_scores = []
				for folder_first_name, post_fix in zip(folder_first_names, post_fixs) : 
					folder_name = folder_first_name + "_" + folder_second_name + "_" + folder_last_name
					stochastic_avgs_10step = []

					avgs = [] 
					times = []
					scores = []
					stddevs = []
					for j in range(random_seed_num):
						loads = [ [0] * controller_num for _ in range(300)]
						rseed = rseed_start + j
						#print(f"{folder_name}/{prefix}burst_{burst}_{switch_num}_{controller_num}_{rseed}{post_fix}.csv")
						file1 = open(f"{folder_name}/{prefix}burst_{burst}_{switch_num}_{controller_num}_{rseed}{post_fix}.csv", 'r') 
						for step in range(300):
							lines = file1.readline().split(",")
							for k in range(controller_num):
								loads[step][k] = float(lines[4*k +3])
						file2 = open(f"{folder_name}/{prefix}burst_{burst}_{switch_num}_{controller_num}_{rseed}{post_fix}_{ep_info}.csv", 'r')
						score = ""
						time = ""
						if(folder_first_name == "nobalance"):
							time =  file2.readline()
						else :
							score_and_time = file2.readline().split(",")
							score = score_and_time[0]
							time = score_and_time[1]
						loads = np.array(loads)
						stddev = np.std(loads, axis=1, ddof=1)
						stddevs.append(stddev)
						avg = np.average(stddev)
						avgs.append(avg)
						times.append(float(time))
						#scores.append(float(score))
					a = np.average(np.array(stddevs), axis=0)
					steps = 0
					for i in range(300-steps):
						if(steps > 0):
							stochastic_avgs_10step.append(np.average(np.array(a[i:i+steps])))
						else :
							stochastic_avgs_10step.append(np.array(a[i]))
					stochastic_avg = sum(avgs) / len(avgs)
					stochastic_avgs.append(stochastic_avg)
					stochastic_times.append(sum(times) / len(times))
					stochastic_avgs_10steps.append(stochastic_avgs_10step)
				print(np.array(stochastic_avgs_10steps).shape)
				with open(f"1221_burst_{burst}_{switch_num}_{controller_num}_{steps}.csv", 'w+', newline='') as csvfile:
					workloadwriter = csv.writer(csvfile, delimiter=',',
					            quotechar='|', quoting=csv.QUOTE_MINIMAL)
					for i in stochastic_avgs_10steps :
						workloadwriter.writerow(i)

	
				list_sto_avgs.append(stochastic_avgs)
				list_sto_times.append(stochastic_times)


				switch_num = switch_num * 2				
				controller_num = controller_num * 2
			with open(f"1221_burst_{burst}_{folder_second_name}_avg.csv", 'w+', newline='') as csvfile:
				workloadwriter = csv.writer(csvfile, delimiter=',',
				            quotechar='|', quoting=csv.QUOTE_MINIMAL)
				for sto_avgs in list_sto_avgs  :
					workloadwriter.writerow(sto_avgs)
			with open(f"1221_burst_{burst}_{folder_second_name}_time.csv", 'w+', newline='') as csvfile:
				workloadwriter = csv.writer(csvfile, delimiter=',',
				            quotechar='|', quoting=csv.QUOTE_MINIMAL)
				for sto_times in list_sto_times  :
					workloadwriter.writerow(sto_times)
