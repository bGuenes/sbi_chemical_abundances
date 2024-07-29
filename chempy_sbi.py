import numpy as np
import sbi.utils

from Chempy.parameter import ModelParameters
import sbi.utils as utils
from sbi.inference.base import infer
import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import time as t


# --- Load the Network ------------------------------------------------------------------------------------------------
# Load network weights trained in train_chempyNN.py
x = np.load('data/tutorial_weights.npz')

w0 = x['w0']
w1 = x['w1']
b0 = x['b0']
b1 = x['b1']
in_mean = x['in_mean']
in_std = x['in_std']
out_mean = x['out_mean']
out_std = x['out_std']
activation = x['activation']
neurons = x['neurons']


# --- Set-up the Simulator --------------------------------------------------------------------------------------------
def add_time_squared(x):
    return np.concatenate((x, (x[:, -1]**2).reshape((len(x), 1))), axis=1)


def stacked_net_output(in_par):
    in_par = (in_par - in_mean) / in_std
    in_par = add_time_squared(in_par)

    l1 = np.matmul(in_par, w0) + b0
    l2 = np.matmul(np.tanh(l1), w1) + b1

    return l2 * out_std + out_mean


# --- Set-up priors ---------------------------------------------------------------------------------------------------
a = ModelParameters()
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])

combined_priors = utils.MultipleIndependent([
    Normal(priors[0, 0]*torch.ones(1), priors[0, 1]*torch.ones(1)),
    Normal(priors[1, 0]*torch.ones(1), priors[1, 1]*torch.ones(1)),
    Normal(priors[2, 0]*torch.ones(1), priors[2, 1]*torch.ones(1)),
    Normal(priors[3, 0]*torch.ones(1), priors[3, 1]*torch.ones(1)),
    Normal(priors[4, 0]*torch.ones(1), priors[4, 1]*torch.ones(1)),
    Uniform(torch.tensor([2.0]), torch.tensor([12.8]))
    ], validate_args=False)

# --- sbi setup -------------------------------------------------------------------------------------------------------
num_sim = 1000
method = 'SNRE' #SNPE or SNLE or SNRE
posterior = infer(
    stacked_net_output,
    combined_priors,
    method=method,
    num_simulations=num_sim)

# --- Save the posterior ----------------------------------------------------------------------------------------------
posterior.save('data/tutorial_posterior')