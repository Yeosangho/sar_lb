
import math
import random
from poisson_process import Switch
import csv
import time
class sim_env():
    def __init__(self, switch_num=20, controller_num=4, rseed=1, burst=False):
        self.burst = burst
        self.rseed = rseed
        #init for test
        random.seed(self.rseed)
        self.switch_num = switch_num
        self.controller_num = controller_num
        self.controller_assigned_info = {}
        for i in range(self.controller_num):
            self.controller_assigned_info[i] = []
        self.switches = []
        self.switches_assigned_info = []
        swtiches_per_ctrl = int(self.switch_num/self.controller_num)

        for i in range(self.switch_num) :
            mu_range = 0
            start_point_range = 0 
            range_min = 0
            range_max = 0
            if(self.burst == False):
                mu_range = random.randint(10, 45)
                start_point_range = random.randint(0, mu_range)
                range_min = random.randint(int(mu_range*0.1), int(mu_range*0.3))
                range_max = mu_range*2 - random.randint(int(mu_range*0.1), int(mu_range*0.3))
            else : 
                mu_range = random.randint(4, 6)
                mu_range_mean = random.randint(40, 48)
                start_point_range = random.randint(int((48/self.controller_num)*(i/swtiches_per_ctrl+0.3)), int((48/self.controller_num)*(i/swtiches_per_ctrl +0.7)))
                range_min = mu_range - mu_range_mean
                range_max = mu_range + mu_range_mean



            base_workload = random.uniform(0,0.1)
            newswitch = Switch(start_point_range,mu_range, range_min, range_max, base_workload)
            self.switches.append(newswitch)
        swtiches_per_ctrl = int(self.switch_num/self.controller_num)
        for i in range(self.controller_num):
            for j in range(swtiches_per_ctrl):
                self.controller_assigned_info[i].extend([j + i*swtiches_per_ctrl])
                self.switches_assigned_info.append(i)
        self.nbs = []
        self.cpus = []
        self.rams = []
        self.utilization  = []
        for switch in self.switches :
            nb, cpu, ram = switch.step()
            self.nbs.append(nb)
            self.cpus.append(cpu)
            self.rams.append(ram)
            self.utilization.append(nb + cpu + ram)

    def reset(self):
        random.seed(self.rseed)
        self.controller_assigned_info = {}
        for i in range(self.controller_num):
            self.controller_assigned_info[i] = []
        self.switches = []
        self.switches_assigned_info = []
        swtiches_per_ctrl = int(self.switch_num/self.controller_num)
        for i in range(self.switch_num):
            mu_range = 0
            start_point_range = 0 
            range_min = 0
            range_max = 0
            if(self.burst == False):
                mu_range = random.randint(10, 45)
                start_point_range = random.randint(0, mu_range)
                range_min = random.randint(int(mu_range*0.1), int(mu_range*0.3))
                range_max = mu_range*2 - random.randint(int(mu_range*0.1), int(mu_range*0.3))
            else : 
                mu_range = random.randint(4, 6)
                mu_range_mean = random.randint(40, 48)
                start_point_range = random.randint(int((48/self.controller_num)*(i/swtiches_per_ctrl+0.3)), int((48/self.controller_num)*(i/swtiches_per_ctrl +0.7)))
                range_min = mu_range - mu_range_mean
                range_max = mu_range + mu_range_mean

            base_workload = random.uniform(0,0.1)
            self.switches.append(Switch(start_point_range,mu_range, range_min, range_max, base_workload))
        
        for i in range(self.controller_num):
            for j in range(swtiches_per_ctrl):
                self.controller_assigned_info[i].extend([j + i*swtiches_per_ctrl])
                self.switches_assigned_info.append(i)
        self.nbs = []
        self.cpus = []
        self.rams = []
        self.utilization  = []
        for switch in self.switches :
            nb, cpu, ram = switch.step()
            self.nbs.append(nb)
            self.cpus.append(cpu)
            self.rams.append(ram)
            self.utilization.append(nb + cpu + ram)

        #make state
        #make state
        state = {}
        state["controllers"] = []
        state["switches"] = []

        for i in range(self.controller_num):
            ctrl_nb = 0
            ctrl_cpu = 0
            ctrl_ram = 0
            for j in self.controller_assigned_info[i] :
                ctrl_nb = ctrl_nb + self.nbs[j]
                ctrl_cpu = ctrl_cpu + self.cpus[j]
                ctrl_ram = ctrl_ram + self.rams[j]
            state["controllers"].extend([i, 0, ctrl_nb, ctrl_cpu, ctrl_ram])


        for i in range(self.switch_num):
            switch_nb = self.nbs[i]
            switch_cpu = self.cpus[i]
            switch_ram = self.rams[i]
            cid = self.switches_assigned_info[i] 
            state["switches"].append([cid, 1, switch_nb, switch_cpu, switch_ram])


        return state

    def step(self, switch_id, controller_id):




            #find src controller in switches
            src_ctrl_id = self.switches_assigned_info[switch_id]
        #if(src_ctrl_id is not controller_id):
            #calc prev utilization difference
            switches_in_src = self.controller_assigned_info[src_ctrl_id]
            src_nb = 0
            src_cpu = 0
            src_ram = 0 
            for i in switches_in_src :
                src_nb = src_nb + self.nbs[i]
                src_cpu = src_cpu + self.cpus[i]
                src_ram = src_ram + self.rams[i]
    
            switches_in_dest = self.controller_assigned_info[controller_id]
            dest_nb = 0
            dest_cpu = 0
            dest_ram = 0 
            for i in switches_in_dest :
                dest_nb = dest_nb + self.nbs[i]
                dest_cpu = dest_cpu + self.cpus[i]
                dest_ram = dest_ram + self.rams[i]
            prev_U_diff = math.pow((src_nb-dest_nb),2) + math.pow((src_cpu-dest_cpu),2) + math.pow((src_ram-dest_ram),2)
    
            #remove switches from 
            self.controller_assigned_info[src_ctrl_id].remove(switch_id)
            #add switch to dest ctrl
            self.controller_assigned_info[controller_id].extend([switch_id])
            #print(self.controller_assigned_info)
            self.switches_assigned_info[switch_id] = controller_id
            
    
            self.nbs = []
            self.cpus = []
            self.rams = []
            self.utilization = []
            for switch in self.switches :
                nb, cpu, ram = switch.step()
                self.nbs.extend([nb])
                self.cpus.extend([cpu])
                self.rams.extend([ram])
                self.utilization.append(nb + cpu + ram)
    
            #calc next utilization difference
            switches_in_src = self.controller_assigned_info[src_ctrl_id]
            src_nb = 0
            src_cpu = 0
            src_ram = 0 
            for i in switches_in_src :
                src_nb = src_nb + self.nbs[i]
                src_cpu = src_cpu + self.cpus[i]
                src_ram = src_ram + self.rams[i]
    
            switches_in_dest = self.controller_assigned_info[controller_id]
            dest_nb = 0
            dest_cpu = 0
            dest_ram = 0 
            for i in switches_in_dest :
                dest_nb = dest_nb + self.nbs[i]
                dest_cpu = dest_cpu + self.cpus[i]
                dest_ram = dest_ram + self.rams[i]
            next_U_diff = math.pow((src_nb-dest_nb),2) + math.pow((src_cpu-dest_cpu),2) + math.pow((src_ram-dest_ram),2)
    
            #calc reward
            reward = prev_U_diff - next_U_diff
            
            #make state
            state = {}
            state["controllers"] = []
            state["switches"] = []
            for i in range(self.controller_num):
                ctrl_nb = 0
                ctrl_cpu = 0
                ctrl_ram = 0
                for j in self.controller_assigned_info[i] :
                    ctrl_nb = ctrl_nb + self.nbs[j]
                    ctrl_cpu = ctrl_cpu + self.cpus[j]
                    ctrl_ram = ctrl_ram + self.rams[j]
                state["controllers"].extend([i, 0, ctrl_nb, ctrl_cpu, ctrl_ram])
            #print(state)
            for i in range(self.switch_num):
                switch_nb = self.nbs[i]
                switch_cpu = self.cpus[i]
                switch_ram = self.rams[i]
                cid = self.switches_assigned_info[i] 
                state["switches"].append([cid, 1, switch_nb, switch_cpu, switch_ram])
    
            return True, state, reward

            
    def update_state_info(self):
            self.nbs = []
            self.cpus = []
            self.rams = []
            self.utilization = []
            for switch in self.switches :
                nb, cpu, ram = switch.step()
                self.nbs.extend([nb])
                self.cpus.extend([cpu])
                self.rams.extend([ram])
                self.utilization.append(nb + cpu + ram)
            #make state
            state = {}
            state["controllers"] = []
            state["switches"] = []
            for i in range(self.controller_num):
                ctrl_nb = 0
                ctrl_cpu = 0
                ctrl_ram = 0
                for j in self.controller_assigned_info[i] :
                    ctrl_nb = ctrl_nb + self.nbs[j]
                    ctrl_cpu = ctrl_cpu + self.cpus[j]
                    ctrl_ram = ctrl_ram + self.rams[j]
                state["controllers"].extend([i, 0, ctrl_nb, ctrl_cpu, ctrl_ram])

            return state["controllers"]      
    def get_state_info(self):
            #make state
            state = {}
            state["controllers"] = []
            state["switches"] = []
            for i in range(self.controller_num):
                ctrl_nb = 0
                ctrl_cpu = 0
                ctrl_ram = 0
                for j in self.controller_assigned_info[i] :
                    ctrl_nb = ctrl_nb + self.nbs[j]
                    ctrl_cpu = ctrl_cpu + self.cpus[j]
                    ctrl_ram = ctrl_ram + self.rams[j]
                state["controllers"].extend([i, 0, ctrl_nb, ctrl_cpu, ctrl_ram])

            return state["controllers"]                  
    def get_obs(self):
            state = {}
            state["controllers"] = []
            state["switches"] = []
            for i in range(self.controller_num):
                ctrl_nb = 0
                ctrl_cpu = 0
                ctrl_ram = 0
                for j in self.controller_assigned_info[i] :
                    ctrl_nb = ctrl_nb + self.nbs[j]
                    ctrl_cpu = ctrl_cpu + self.cpus[j]
                    ctrl_ram = ctrl_ram + self.rams[j]
                state["controllers"].extend([i, 0, ctrl_nb, ctrl_cpu, ctrl_ram])
            #print(state)
            for i in range(self.switch_num):
                switch_nb = self.nbs[i]
                switch_cpu = self.cpus[i]
                switch_ram = self.rams[i]
                cid = self.switches_assigned_info[i] 
                state["switches"].append([cid, 1, switch_nb, switch_cpu, switch_ram])
            return state           
        #else :
        #    return False, state, reward
if __name__ == '__main__':
    for out_out_iter in range(2):
        times = 1
        burst=True
        if(out_out_iter == 0) :
            burst = False
        
        for out_iter in range(4):
            for in_iter in range(4):
                controller_num = 4 * 1
                switch_num = 20 * times
                rseed=777 + in_iter
                env = sim_env(switch_num, controller_num, rseed, burst)
                env.reset()
                start_time = time.time()
                rows = []
                for i in range(300):
                    env.nbs = []
                    env.cpus = []
                    env.rams = []
                    env.utilization = []
                    for switch in env.switches :
                        nb, cpu, ram = switch.step()
                        env.nbs.extend([nb])
                        env.cpus.extend([cpu])
                        env.rams.extend([ram])
                        env.utilization.append(nb + cpu + ram)
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
                with open(f'after_burst_{burst}_{switch_num}_{controller_num}_{rseed}_noloadbalance_ep_info.csv', 'w+', newline='') as f:    
                    print(f"{time.time()-start_time}", file=f)                
                #with open(f'after_burst_{burst}_{switch_num}_{controller_num}_{rseed}_noloadbalance.csv', 'w+', newline='') as csvfile:    
                #    workloadwriter = csv.writer(csvfile, delimiter=',',
                #                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                #    for i in range(300):
                #        workloadwriter.writerow(rows[i])
            times = times * 2
        
