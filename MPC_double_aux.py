import sys
import os
import yaml

from model_predictive_controller import *

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(os.path.join(ROOT_DIR, 'masters_project', 'app', 'CBcurl_master', 'CBcurl'))

from utilities import *

# open parameter file
#f = open('/home/neythen/Desktop/Projects/masters_project/app/CBcurl_master/examples/parameter_files/smaller_target.yaml')
f = open('/home/neythen/Desktop/Projects/masters_project/app/CBcurl_master/examples/parameter_files/MPC.yaml')
param_dict = yaml.load(f)
f.close()

param_dict = convert_to_numpy(param_dict)

ode_params = param_dict['ode_params']
initial_X = param_dict['Q_params'][7]
initial_Cs = param_dict['Q_params'][8]
initial_C0 = param_dict['Q_params'][9]

initial_S = np.append(initial_X, initial_Cs)
initial_S = np.append(initial_S, initial_C0)


Cins_domain = np.array([[0, 0.1], [0, 0.1]])

target = np.array([[20000.,10000.]]*10)



n_particles = 10
n_groups = 3

optimiser_args = [Cins_domain, n_particles, n_groups, [2,2], ode_params]

MPC = MPC(*optimiser_args)

n_timesteps = 500
n_opt_steps = 20

params = ode_params
print(initial_S)
MPC.run(initial_S, params, target, n_timesteps, n_opt_steps)
TS = np.array(MPC.time_series)[:,:2]
print(MPC.time_series[-1])
plt.plot(TS, label = 'population curve')
plt.xlabel('Time (hours)')
plt.ylabel('Population (cells/L)')
plt.hlines(y = target[0], xmin = 0, xmax = n_timesteps, label = 'target', linestyle = '--')
plt.hlines(y = target[1], xmin = 0, xmax = n_timesteps, linestyle = '--')
plt.legend()
plt.show()
