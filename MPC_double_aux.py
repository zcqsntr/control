import sys
import os
import yaml

from model_predictive_controller import *

sys.path.append('/Users/Neythen/Desktop/masters_project/app/CBcurl_master/CBcurl')

from utilities import *

# open parameter file
f = open('/Users/Neythen/Desktop/masters_project/app/CBcurl_master/examples/parameter_files/smaller_target.yaml')
param_dict = yaml.load(f)
f.close()

param_dict = convert_to_numpy(param_dict)

ode_params = param_dict['ode_params']
initial_X = param_dict['Q_params'][7]
initial_Cs = param_dict['Q_params'][8]
initial_C0 = param_dict['Q_params'][9]

initial_S = np.append(initial_X, initial_Cs)
initial_S = np.append(initial_S, initial_C0)

print(initial_S)
Cins_domain = np.array([[0, 0.1], [0, 0.1]])

target = np.array([[250., 550.]]*10)

print(target.shape)

n_particles = 10
n_groups = 3

optimiser_args = [Cins_domain, n_particles, n_groups, [2,2], ode_params]

MPC = MPC(*optimiser_args)

n_timesteps = 50
n_opt_steps = 40
params = ode_params[2:]
print(ode_params)

MPC.run(initial_S, params, target, n_timesteps, n_opt_steps)
TS = np.array(MPC.time_series)[:,:2]
plt.plot(TS, label = 'population curve')
plt.xlabel('Time (hours)')
plt.ylabel('Population (10^6 cells/L)')
plt.hlines(y = 250., xmin = 0, xmax = n_timesteps, label = 'target', linestyle = '--')
plt.hlines(y = 550., xmin = 0, xmax = n_timesteps, linestyle = '--')
plt.legend()
plt.show()
