import sys
import os
import yaml


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.join(ROOT_DIR, 'parameter_estimation', 'global_optimsiation'))


from hybrid_PS_SGD_v2 import *

class MPC():

    def __init__(self, domain, n_particles, n_groups, cs, velocity_scaling, ode_params):
        self.swarm = Swarm(domain, n_particles, n_groups, cs, velocity_scaling, ode_params)
        self.time_series = []

    def reset_swarm(self,domain, n_particles, n_groups, cs, velocity_scaling, ode_params):
        self.swarm = Swarm(domain, n_particles, n_groups, cs, velocity_scaling, ode_params)

    def run(self, initial_S, params, target, n_timesteps, n_steps):
        S = initial_S
        for i in range(n_timesteps):

            print('i: ', i)
            print(S)
            losses, particle_positions, _ = self.swarm.find_minimum(S, params, target, n_steps, 'MPC')

            argmin = np.argmin(losses)
            best_Cin = particle_positions[argmin]
            print('position: ', best_Cin)
            print('loss: ', min(losses))
            S = self.swarm.predict(params, S, best_Cin, [0,1])[0]
            self.time_series.append(S)



# open parameter file
f = open('../parameter_estimation/parameter_files/monoculture.yaml')
param_dict = yaml.load(f)
f.close()

ode_params = param_dict['ode_params']
initial_X = param_dict['Q_params'][6]
initial_C0 = param_dict['Q_params'][7]

initial_S = np.append(initial_X, initial_C0)

Cin_domain = np.array([[0, 1.5]])

target = np.array([30000]*10) # target for MPC to aim for, steady state of 300000


parameters = np.array([480000, 0.6])
n_particles = 10
n_groups = 1
optimiser_args = [Cin_domain, n_particles, n_groups, [2,2], np.array([1]), ode_params]


MPC = MPC(*optimiser_args)
n_timesteps = 40
n_steps = 20
MPC.run(initial_S, parameters, target, n_timesteps, n_steps)


target = np.array([35000]*10)
MPC.reset_swarm(*optimiser_args)
MPC.run(MPC.time_series[-1], parameters, target, n_timesteps, n_steps)
TS = np.array(MPC.time_series)[:,0]
plt.plot(TS, label = 'population curve')
plt.xlabel('Time (hours)')
plt.ylabel('Population (10^6 cells/L)')
plt.hlines(y = 30000, xmin = 0, xmax = n_timesteps, label = 'target', linestyle = '--')
plt.hlines(y = 35000, xmin = n_timesteps, xmax = n_timesteps*2, linestyle = '--')
plt.legend()
plt.show()
