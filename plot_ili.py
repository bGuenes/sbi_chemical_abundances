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

name = "NPE_C_nsf_5sigma_uni_prior"

# ----- Load the data -----
a = ModelParameters()
labels = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])

elements = a.elements_to_trace

# ----- Load posterior -----
with open(f'data/posterior_{name}.pickle', 'rb') as f:
    posterior = pickle.load(f)

# ----- Load the data ---------------------------------------------------------------------------------------------------------------------------------------------
# ----- Evaluate the posterior -------------------------------------------------------------------------------------------------------------------------------------------
# --- Load the validation data ---
# Validation data created with CHEMPY, not with the NN simulator
print("Evaluating the posterior...")
file_path = os.path.dirname(__file__)
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
pc_ab = 5
x_err = np.ones_like(abundances)*float(pc_ab)/100.
abundances = norm.rvs(loc=abundances,scale=x_err)
abundances = torch.tensor(abundances).float()

labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']

# --- Plot calbration using ltu-ili ---
from metrics import PosteriorCoverage

plot_hist = ["coverage", "histogram", "predictions", "tarp"]
metric = PosteriorCoverage(
    num_samples=1000, sample_method="direct",
    labels=labels_in,
    plot_list = plot_hist
)

fig = metric(
    posterior=posterior,
    x=abundances, theta=val_theta)

for i, plot in enumerate(fig):
    fig[i].savefig(file_path+ f"/plots/ili_{plot_hist[i]}.pdf")