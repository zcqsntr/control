import sys
import os
import yaml


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(ROOT_DIR, 'masters_project', 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.join(ROOT_DIR, 'parameter_estimation', 'global_optimsiation'))


from hybrid_PS_SGD_v2 import *

class MPC():

    def __init__(self, domain, n_particles, n_groups, cs, ode_params):
        self.swarm = Swarm(domain, n_particles, n_groups, cs, ode_params)
        self.time_series = []

    def reset_swarm(self,domain, n_particles, n_groups, cs, ode_params):
        self.swarm = Swarm(domain, n_particles, n_groups, cs,  ode_params)

    def run(self, initial_S, params, target, n_timesteps, n_steps):
        print(params)
        noisey_params = [0.25, 0.5, np.array([520000, 440000]), np.array([400000, 560000]), np.array([1.7, 2.2]), np.array([0.00053, 0.00000113]), np.array([0.000074, 0.000064]), np.array([[0,0], [0,0]])] # about 5% error on pump rate, 10% on other params
        S = initial_S
        self.time_series.append(initial_S)
        for i in range(n_timesteps):
            print(i)
            losses, particle_positions, _ = self.swarm.find_minimum(S, noisey_params, target, n_steps, 'MPC')

            argmin = np.argmin(losses)
            best_Cin = particle_positions[argmin]
            print(best_Cin)

            # add 5% noise to actions and state measurement
            best_Cin = np.random.normal(best_Cin, 0.05 * best_Cin, size = (2,))

            S = self.swarm.predict(params, S, best_Cin, [0,1])[0]
            print(S)
            S = np.random.normal(S, 0.05 * S, size = (5,))

            self.time_series.append(S)
