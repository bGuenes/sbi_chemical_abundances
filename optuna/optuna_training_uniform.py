import numpy as np

from scipy.stats import norm

from Chempy.parameter import ModelParameters

import sbi.utils as utils
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import simulate_for_sbi, NPE_C
from sbi.neural_nets import posterior_nn
from metrics import PosteriorSamples
import tarp

import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import time as t
import pickle
from tqdm import tqdm


import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState


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
    

def objective(trial):
    
        
    # model = trial.suggest_categorical('model', ['nsf', 'maf', 'maf_rqs'])    
    hidden_features = trial.suggest_categorical('hidden_features', [10, 20, 50, 100])
    num_transforms = trial.suggest_categorical('num_transforms', [1, 5, 10, 30])
    
    # ----- Train the SBI -------------------------------------------------------------------------------------------------------------------------------------------
    density_estimator_build_fun = posterior_nn(model='nsf', hidden_features=hidden_features, num_transforms=num_transforms, )
    inference = NPE_C(prior=prior, density_estimator=density_estimator_build_fun, show_progress_bars=True, device=device)

    # --- train ---
    density_estimator = inference.append_simulations(theta, x).train(show_train_summary=True, training_batch_size=10_000, max_num_epochs=2)

    # --- build the posterior ---
    posterior = inference.build_posterior(density_estimator)

    nll_test = -torch.mean(posterior.log_prob_batched(theta_test.repeat(1000, 1).view(1000, 1000, 6), x=x_test) )



    sampler = PosteriorSamples(num_samples=500, sample_method='direct')
    xv, tv = x_test.to(device), theta_test.to(device)
    samps = sampler(posterior, xv, tv)

    # measure tarp
    ecp, alpha = tarp.get_tarp_coverage(
        samps, tv.cpu().numpy(),
        norm=True, bootstrap=True,
        num_bootstrap=100
    )

    tarp_val = torch.mean(torch.from_numpy(ecp[:,ecp.shape[1]//2])).to(device)

    
    return nll_test, abs(tarp_val-0.5)


if __name__ == '__main__':
    
    device='cuda:3'

    # ----- Config -------------------------------------------------------------------------------------------------------------------------------------------

    name = "NPE_C"

    # ----- Load the model -------------------------------------------------------------------------------------------------------------------------------------------
    # --- Define the prior ---
    a = ModelParameters()
    labels_out = a.elements_to_trace
    labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']
    priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])

    combined_priors = utils.MultipleIndependent(
        [Normal(p[0]*torch.ones(1, device=device), p[1]*torch.ones(1, device=device)) for p in priors] +
        [Uniform(torch.tensor([2.0], device=device), torch.tensor([12.8], device=device))],
        validate_args=False)
    
    # combined_priors = utils.MultipleIndependent(
    # [Uniform(p[0]*torch.ones(1)-5*p[1], p[0]*torch.ones(1)+5*p[1]) for p in priors] +
    # [Uniform(torch.tensor([2.0]), torch.tensor([12.8]))],
    # validate_args=False)

    prior, num_parameters, prior_returns_numpy = process_prior(combined_priors)
    
    training_data = np.load('data/optuna/training_uni.npz')
    x = torch.tensor(training_data['x']).to(device)
    theta =  torch.tensor(training_data['theta']).to(device)
    
    test_data = np.load('data/optuna/test_uni.npz')
    x_test = torch.tensor(test_data['x_test']).to(device)
    theta_test = torch.tensor(test_data['theta_test']).to(device)

    study_name = 'example_study_nsf_uni'  # Unique identifier of the study.
    storage_name = 'sqlite:///example_onlylog_nsf_uni.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name,directions=['maximize', 'minimize'], load_if_exists=True)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, callbacks=[MaxTrialsCallback(16, states=(TrialState.COMPLETE, TrialState.FAIL))],)
    


