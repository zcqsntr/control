import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr


def linear_model(x,u, A, B):

    sdot = np.matmul(A, x) + np.matmul(B, u)

    return sdot


def predict(x, t, u, A, B):
    # extract params from param_vec into form used by the rest of the code
    time_diff = 2  # frame skipping
    time_points = np.array([x *1 for x in range(time_diff)])

    sol = odeint(linear_model, N, time_points, tuple((u, A, B)))[1:]

    pred_x = sol[-1, 0:2]

    return pred_x

def objective(current_x, next_x, u, A, B, Q, R):
    x_pred = predict(current_x, u, A, B)
    return np.matmul((x_pred - next_x).T, np.matmul(Q, x_pred - next_x)) + np.matul(u.T, np.matmul(R, u))



grad_func = grad(objective)

def grad_wrapper(param_vec, i):
    '''
    for the autograd optimisers

    param_vec = [flatten(A), flatten(B)]
    '''

    A = param_vec[:4].reshape(2,2)
    B = param_vec[4:].reshape(2,2)


    return grad_func(current_x, next_x, u, A, B, Q, R)




param_vec = np.array([-0.1, -0.1, -0.1, -0.1, 1, 0, 0, 1])

for i in range(T_MAX):
    # get measured x
    # predict for time horizon
