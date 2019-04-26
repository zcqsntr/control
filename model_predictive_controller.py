import sys
import os
import yaml


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(ROOT_DIR, 'masters_project', 'app', 'CBcurl_master', 'CBcurl'))
sys.path.append(os.path.join(ROOT_DIR, 'parameter_estimation', 'global_optimsiation'))

print(ROOT_DIR)
from particle_swarm_optimiser import *
from hybrid_oxford import OxfordSystem



class MPC():

    def __init__(self, domain, n_particles, n_groups, cs, ode_params):
        self.optimiser = Swarm(OxfordSystem(ode_params), domain, n_particles, n_groups, cs, ode_params)
        self.time_series = []

    def reset_swarm(self, domain, n_particles, n_groups, cs, ode_params):
        self.optimiser = Swarm(domain, n_particles, n_groups, cs,  ode_params)

    def run(self, initial_S, params, target, n_timesteps, n_steps):
        print(params)
        noisey_params = [0.25, 0.5, np.array([520000, 440000]), np.array([400000, 560000]), np.array([1.7, 2.2]), np.array([0.00053, 0.00000113]), np.array([0.000074, 0.000064]), np.array([[0,0], [0,0]])] # about 5% error on pump rate, 10% on other params
        predicted_S = initial_S
        S = initial_S
        self.time_series.append(initial_S)

        for i in range(n_timesteps):
            print(i)
            losses, particle_positions, _ = self.optimiser.find_minimum(predicted_S, noisey_params, target, n_steps, 'MPC')

            argmin = np.argmin(losses)
            best_Cin = particle_positions[argmin]
            print(best_Cin)

            # add 5% noise to actions and state measurement
            best_Cin = np.random.normal(best_Cin, 0.05 * best_Cin, size = (2,))

            S = self.optimiser.system.predict(params, S, best_Cin, [0,1])[0]
            predicted_S = self.optimiser.system.predict(noisey_params, predicted_S, best_Cin, [0,1])[0]
            print(predicted_S)
            N = S[0:2]

            N = np.random.normal(N, 0.05 * N, size = (2,))
            predicted_S[0] = N[0]
            predicted_S[1] = N[1]
            print(predicted_S)
            self.time_series.append(predicted_S)





class linear_MPC():

    def __init__(self, actual_params, A, B):
        self.optimiser = GradientDescentOptimiser(Q, R)
        self.non_linear_model = Swarm(np.array([[0,1], [0,1]]), 1, 1, [2,2], actual_params) # to predict the actual next state using the full model
        self.time_series = []


    def run(self, initial_S, initial_linear_params, actual_params, target, n_timesteps, n_steps):
        # S is the full state use to simulate actual system
        # x is the current population which is used by the linear model

        current_S = initial_S
        current_x = initial_S[0:2]
        self.time_series.append(initial_x)
        linear_params = initial_linear_params

        for i in range(n_timesteps):
            # optimise wrt control action
            next_Cin = self.optimiser.find_minimum(current_x, linear_param_vec, target, n_steps, 'MPC')

            next_S = self.non_linear_model.predict(self.actual_params, current_S, Cin, [0,1])[0]
            next_x = current_S[0:2]

            # optimise wrt the linear model
            linear_params = self.optimiser.find_minimum(current_x, Cin, next_x, n_steps, 'param_est')

            self.time_series.append(next_S)
