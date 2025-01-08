import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Chempy.parameter import ModelParameters

import sbi.utils as utils
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import NPE, simulate_for_sbi, NPE_C
from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp
from sbi.neural_nets import posterior_nn

import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import time as t
import pickle
import os
from tqdm import tqdm

from plot_functions import *

# ----- Config -------------------------------------------------------------------------------------------------------------------------------------------

file_path = os.path.dirname(__file__)
name = "NPE_C"

# ----- Load the model -------------------------------------------------------------------------------------------------------------------------------------------
# --- Define the prior ---
a = ModelParameters()
labels_out = a.elements_to_trace
labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]*2] for opt in a.to_optimize])

combined_priors = utils.MultipleIndependent(
    [Normal(p[0]*torch.ones(1), p[1]*torch.ones(1)) for p in priors] +
    [Uniform(torch.tensor([2.0]), torch.tensor([12.8]))],
    validate_args=False)


# --- Set up the model ---
class Model_Torch(torch.nn.Module):
    def __init__(self):
        super(Model_Torch, self).__init__()
        self.l1 = torch.nn.Linear(len(labels_in), 100)
        self.l2 = torch.nn.Linear(100, 40)
        self.l3 = torch.nn.Linear(40, len(labels_out))

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x

model = Model_Torch()

# --- Load the weights ---
model.load_state_dict(torch.load(file_path + '/data/pytorch_state_dict.pt'))
model.eval()


# ----- Set up the simulator -------------------------------------------------------------------------------------------------------------------------------------------
def simulator(params):
    y = model(params)
    y = y.detach().numpy()

    # Remove H from data, because it is just used for normalization (output with index 2)
    y = np.delete(y, 2)

    return y


prior, num_parameters, prior_returns_numpy = process_prior(combined_priors)
simulator = process_simulator(simulator, prior, prior_returns_numpy)
check_sbi_inputs(simulator, prior)


# ----- Train the SBI -------------------------------------------------------------------------------------------------------------------------------------------
density_estimator_build_fun = posterior_nn(model="maf", hidden_features=50, num_transforms=5)
inference = NPE_C(prior=prior, density_estimator=density_estimator_build_fun, show_progress_bars=True)

start = t.time()

# --- simulate the data ---
print()
print("Simulating data...")
theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=500_000)
print(f"Genereted {len(theta)} samples")

# --- add noise ---
pc_ab = 5 # percentage error in abundance

x_err = np.ones_like(x)*float(pc_ab)/100.
x = norm.rvs(loc=x,scale=x_err)
x = torch.tensor(x).float()

# --- train ---
print()
print("Training the posterior...")
density_estimator = inference.append_simulations(theta, x).train(show_train_summary=True)

# --- build the posterior ---
posterior = inference.build_posterior(density_estimator)

end = t.time()
comp_time = end - start

print()
print(f'Time taken to train the posterior with {len(theta)} samples: '
      f'{np.floor(comp_time/60).astype("int")}min {np.floor(comp_time%60).astype("int")}s')


# ----- Save the posterior -------------------------------------------------------------------------------------------------------------------------------------------
with open(f'data/posterior_{name}.pickle', 'wb') as f:
    pickle.dump(posterior, f)

print()
print("Posterior trained and saved!")
print()


# ----- Evaluate the posterior -------------------------------------------------------------------------------------------------------------------------------------------
# --- Load the validation data ---
# Validation data created with CHEMPY, not with the NN simulator
print("Evaluating the posterior...")
path_test = file_path + '/data/chempy_data/chempy_TNG_val_data.npz'
val_data = np.load(path_test, mmap_mode='r')

val_theta = val_data['params']
val_x = val_data['abundances']


# --- Clean the data ---
# Chempy sometimes returns zeros or infinite values, which need to removed
def clean_data(x, y):
    # Remove all zeros from the training data
    index = np.where((y == 0).all(axis=1))[0]
    x = np.delete(x, index, axis=0)
    y = np.delete(y, index, axis=0)

    # Remove all infinite values from the training data
    index = np.where(np.isfinite(y).all(axis=1))[0]
    x = x[index]
    y = y[index]

    return x, y

val_theta, val_x = clean_data(val_theta, val_x)

# convert to torch tensors
val_theta = torch.tensor(val_theta, dtype=torch.float32)
val_x = torch.tensor(val_x, dtype=torch.float32)
abundances =  torch.cat([val_x[:,:2], val_x[:,3:]], dim=1)

# add noise to data to simulate observational errors
x_err = np.ones_like(abundances)*float(pc_ab)/100.
abundances = norm.rvs(loc=abundances,scale=x_err)
abundances = torch.tensor(abundances).float()

theta_hat = torch.zeros_like(val_theta)
for index in tqdm(range(len(abundances))):
    thetas_predicted = posterior.sample((1000,), x=abundances[index], show_progress_bars=False)
    theta_predicted = thetas_predicted.mean(dim=0)
    theta_hat[index] = theta_predicted

ape = torch.abs((val_theta - theta_hat) / val_theta) *100
torch.save(ape, f'data/ape_posterior_{name}.pt')


# --- Absolute percentage error plot ---

save_path = file_path + f'/plots/ape_posterior_{name}.png'
ape_plot(ape, labels_in, save_path)

# --- Simulation based calibration plot ---

def simulator(params):
    y = model(params)
    y = y.detach().numpy()

    # Remove H from data, because it is just used for normalization (output with index 2)
    y = np.delete(y, 2,1)

    return y

num_sbc_samples = 200  # choose a number of sbc runs, should be ~100s
# generate ground truth parameters and corresponding simulated observations for SBC.
thetas = combined_priors.sample((num_sbc_samples,))
xs = simulator(thetas)

# run SBC: for each inference we draw 1000 posterior samples.
num_posterior_samples = 1_000
num_workers = 1
ranks, dap_samples = run_sbc(
    thetas, xs, posterior, num_posterior_samples=num_posterior_samples, num_workers=num_workers
)

f, ax = sbc_rank_plot(
    ranks=ranks,
    num_posterior_samples=num_posterior_samples,
    parameter_labels=labels_in,
    plot_type="hist",
    num_cols=3,
    figsize=(15,10),
    num_bins=None,  # by passing None we use a heuristic for the number of bins.
)

f.suptitle("SBC rank plot", fontsize=36)
plt.tight_layout()
plt.savefig(file_path + f'/plots/sbc_rank_plot_{name}.png')
plt.clf()