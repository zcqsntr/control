import sys
import os
import yaml


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(ROOT_DIR, 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.join(ROOT_DIR, 'parameter_estimation', 'global_optimsiation'))


from hybrid_PS_SGD_v2 import *

class MPC():

    def __init__(self, domain, n_particles, n_groups, cs, ode_params):
        self.swarm = Swarm(domain, n_particles, n_groups, cs, ode_params)
        self.time_series = []

    def reset_swarm(self,domain, n_particles, n_groups, cs, ode_params):
        self.swarm = Swarm(domain, n_particles, n_groups, cs,  ode_params)

    def run(self, initial_S, params, target, n_timesteps, n_steps):
        S = initial_S
        for i in range(n_timesteps):
            print(i)


            losses, particle_positions, _ = self.swarm.find_minimum(S, params, target, n_steps, 'MPC')

            argmin = np.argmin(losses)
            best_Cin = particle_positions[argmin]

            S = self.swarm.predict(params, S, best_Cin, [0,1])[0]
            self.time_series.append(S)
