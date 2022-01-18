# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from scipy.stats import poisson
import random

##------------------------------------------------------------
## Define the distribution parameters to be plotted
#mu_values = [15, 15, 15]
#times = [1, 1.2, 0.5]
#linestyles = ['-', '--', ':']
#
##------------------------------------------------------------
## plot the distributions
##   we generate it using scipy.stats.poisson().  Once the distribution
##   object is created, we have many options: for example
##   - dist.pmf(x) evaluates the probability mass function in the case of
##     discrete distributions.
##   - dist.pdf(x) evaluates the probability density function for
##   evaluates
#fig, ax = plt.subplots(figsize=(5, 3.75))
#
#for time, mu, ls in zip(times, mu_values, linestyles):
#    # create a poisson distribution
#    # we could generate a random sample from this distribution using, e.g.
#    #   rand = dist.rvs(1000)
#    dist = poisson(mu)
#    x = np.arange(3, 27)
#
#    for i in range(len(x)):
#    	print(f"{x[i]}, {time*dist.pmf(x[i])}")
#    #plt.plot(x, dist.pmf(x), color='black',
#    #         linestyle='steps-mid' + ls,
#    #         label=r'$\mu=%i$' % mu)

#plt.xlim(-0.5, 30)
#plt.ylim(0, 0.4)
#
#plt.xlabel('$x$')
#plt.ylabel(r'$p(x|\mu)$')
#plt.title('Poisson Distribution')
#
#plt.legend()
#plt.show()

class Switch():
    def __init__(self, start_point=0, mu=30, range_min=0, range_max=60, base_workload=0):
        self.mu = mu
        self.base_workload = base_workload
        self.start_point = start_point
        self.range_max = range_max
        self.range_min = range_min
        self.x_range = np.arange(self.range_min, self.range_max)

        self.dist = poisson(self.mu)

        self.step_count = self.start_point
        
        self.workload_max = 1.35
        self.workload_min = 0.65
##
        self.cpu_max = 1.2
        self.cpu_min = 0.6
##
        self.ram_max = 1.2
        self.ram_min = 0.8

        #self.workload_max = 1.00
        #self.workload_min = 1.00
####
        #self.cpu_max = 1.0
        #self.cpu_min = 1.0
####
        #self.ram_max = 1.0
        #self.ram_min = 1.0


    def step(self):
        workload = self.dist.pmf(self.x_range[self.step_count]) 
        workload = (workload + self.base_workload) 
        nb_workload = workload * random.uniform(self.workload_min, self.workload_max)
        cpu_workload = nb_workload * random.uniform(self.cpu_min, self.cpu_max)
        ram_workload = cpu_workload * random.uniform(self.ram_min, self.ram_max)
        self.step_count = self.step_count + 1
        self.step_count  = self.step_count % len(self.x_range)




        return nb_workload, cpu_workload, ram_workload

if __name__ == '__main__':
    switches = []
    for i in range(4):
        mu_range = random.randint(4, 4)
        start_point_range = random.randint(0,4)
        range_min = 4 - 13
        range_max = 4 + 13
        #range_min = random.randint(int(mu_range*0.1), int(mu_range*0.3))
        #range_max = mu_range*2 - random.randint(int(mu_range*0.1), int(mu_range*0.3))
        base_workload = random.uniform(0,0.1)
        switches.append(Switch(start_point_range,mu_range, range_min, range_max, base_workload))
    #switch2 = switch(15, 30, 7, 53)
    #switch3 = switch(2, 10, 3, 17)
    for i in range(200):
        nb = 0
        cpu = 0
        ram = 0
        for i in range(4):
            nb1,cpu1, ram1 = switches[i].step()
            nb = nb + nb1
            cpu = cpu + cpu1
            ram = ram + ram1
        #nb2,cpu2, ram2 = switch2.step()
        #nb3,cpu3, ram3 = switch3.step()
        #nb = nb1 + nb2 + nb3
        #cpu = cpu1 + cpu2 + cpu3
        #ram = ram1 + ram2 + ram3
        print(f"{nb},{cpu},{ram}")
