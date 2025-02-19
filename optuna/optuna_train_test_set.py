import numpy as np

from scipy.stats import norm

from Chempy.parameter import ModelParameters

import sbi.utils as utils
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import simulate_for_sbi, NPE_C
from sbi.neural_nets import posterior_nn

import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import time as t
import pickle
from tqdm import tqdm

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState



# ----- Set up the simulator -------------------------------------------------------------------------------------------------------------------------------------------
def simulator(params):
    y = model(params)
    # y = y.detach().numpy()

    # Remove H from data, because it is just used for normalization (output with index 2)
    y = torch.cat((y[:2], y[3:]))

    return y

class Model_Torch(torch.nn.Module):
    def __init__(self, x_shape, y_shape):
        super(Model_Torch, self).__init__()
        self.l1 = torch.nn.Linear(x_shape, 100)
        self.l2 = torch.nn.Linear(100, 40)
        self.l3 = torch.nn.Linear(40, y_shape)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x


if __name__ == '__main__':
    
    device='cuda:7'

    # ----- Config -------------------------------------------------------------------------------------------------------------------------------------------

    name = "NPE_C"

    # ----- Load the model -------------------------------------------------------------------------------------------------------------------------------------------
    # --- Define the prior ---
    a = ModelParameters()
    labels_out = a.elements_to_trace
    labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']
    priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])

    # combined_priors = utils.MultipleIndependent(
    #     [Normal(p[0]*torch.ones(1, device=device), p[1]*torch.ones(1, device=device)) for p in priors] +
    #     [Uniform(torch.tensor([2.0], device=device), torch.tensor([12.8], device=device))],
    #     validate_args=False)
    
    combined_priors = utils.MultipleIndependent(
    [Uniform(p[0]*torch.ones(1,device=device)-5*p[1], p[0]*torch.ones(1,device=device)+5*p[1]) for p in priors] +
    [Uniform(torch.tensor([2.0],device=device), torch.tensor([12.8],device=device))],
    validate_args=False)

    # --- Set up the model ---
    model = Model_Torch(len(labels_in), len(labels_out))

    # --- Load the weights ---
    model.load_state_dict(torch.load('data/pytorch_state_dict_5sigma_uni_prior.pt'))
    model.eval()
    model = model.to(device=device)

    prior, num_parameters, prior_returns_numpy = process_prior(combined_priors)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)


    # --- simulate the data ---
    num_sim_train = 100_000
    print()
    print("Simulating data...")
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sim_train)
    print(f"Genereted {len(theta)} samples")

    #training set
    # --- add noise ---
    pc_ab = 5 # percentage error in abundance

    x_err = np.ones_like(x.cpu().detach())*float(pc_ab)/100.
    x = norm.rvs(loc=x.cpu().detach(),scale=x_err)
    x = torch.tensor(x).float().to(device=device)

    #test set
    num_sim_test = 1_000
    theta_test, x_test = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sim_test)
    
    # --- add noise ---
    pc_ab = 5 # percentage error in abundance
    x_err = np.ones_like(x_test.cpu().detach())*float(pc_ab)/100.
    x_test = norm.rvs(loc=x_test.cpu().detach(),scale=x_err)
    x_test = torch.tensor(x_test).float().to(device=device)

    
    np.savez(file='data/optuna/training_uni.npz',
            x=np.array(x.cpu()),
            theta=np.array(theta.cpu()))
    
    np.savez(file='data/optuna/test_uni.npz',
            x_test=x_test.cpu(),
            theta_test=theta_test.cpu())