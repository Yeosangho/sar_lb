
import math
import random
from poisson_process import Switch

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
        state = []

        for i in range(self.controller_num):
            ctrl_nb = 0
            ctrl_cpu = 0
            ctrl_ram = 0
            for j in self.controller_assigned_info[i] :
                ctrl_nb = ctrl_nb + self.nbs[j]
                ctrl_cpu = ctrl_cpu + self.cpus[j]
                ctrl_ram = ctrl_ram + self.rams[j]
            state.extend([i, 0, ctrl_nb, ctrl_cpu, ctrl_ram])


        return state

    def step(self, switch_id, controller_id):



        if(switch_id is not -1):
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
        else :
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
            #print(state)
            for i in range(self.switch_num):
                switch_nb = self.nbs[i]
                switch_cpu = self.cpus[i]
                switch_ram = self.rams[i]
                cid = self.switches_assigned_info[i] 
                state["switches"].append([cid, 1, switch_nb, switch_cpu, switch_ram])
    
            return False, state, 0
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