import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Chempy.parameter import ModelParameters

import sbi.utils as utils
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import NPE, simulate_for_sbi
from sbi.analysis.plot import sbc_rank_plot
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp

import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import time as t
import pickle
import os
import tqdm

file_path = os.path.dirname(__file__)

# ----- Load the model -------------------------------------------------------------------------------------------------------------------------------------------
# --- Define the prior ---
a = ModelParameters()
labels_out = a.elements_to_trace
labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])

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
inference = NPE(prior=prior, show_progress_bars=True)

start = t.time()

# --- simulate the data ---
print()
print("Simulating data...")
theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=1_000_000)
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
with open('data/posterior_sbi_w5p-error_noH.pickle', 'wb') as f:
    pickle.dump(posterior, f)

print()
print("Posterior trained and saved!")
print()


# ----- Evaluate the posterior -------------------------------------------------------------------------------------------------------------------------------------------
# --- Load the validation data ---

print("Evaluating the posterior...")
path_test = file_path + '/data/chempy_data/TNG_Test_Data.npz'
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

val_theta, val_x     = clean_data(val_theta, val_x)

# convert to torch tensors
val_theta = torch.tensor(val_theta, dtype=torch.float32)
val_x = torch.tensor(val_x, dtype=torch.float32)
abundances =  torch.cat([val_x[:,:2], val_x[:,3:]], dim=1)

# add noise to data
x_err = np.ones_like(abundances)*float(pc_ab)/100.
abundances = norm.rvs(loc=abundances,scale=x_err)
abundances = torch.tensor(abundances).float()

theta_hat = torch.zeros_like(val_theta)
for index in tqdm(range(len(abundances))):
    thetas_predicted = posterior.sample((1000,), x=abundances[index], show_progress_bars=False)
    theta_predicted = thetas_predicted.mean(dim=0)
    theta_hat[index] = theta_predicted

ape = torch.abs((val_theta - theta_hat) / val_theta) *100


# --- Absolute percentage error plot ---

fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)})
colors = ["tomato", "skyblue", "olive", "gold", "teal", "orchid"]

for i in range(6):
    l_quantile, median, u_quantile = np.percentile(ape[:,i], [25, 50, 75])
    ax_hist.hist(ape[:,i], bins=25, density=True, range=(0, 100), label=labels_in[i], color=colors[i], alpha=0.5)
    median = np.percentile(ape[:,i], 50)
    ax_hist.axvline(median, color=colors[i], linestyle='--')
    print(labels_in[i] + f" : {median:.1f}% + {u_quantile-median:.1f} - {median-l_quantile:.1f}")
    print()
    
ax_hist.set_xlabel('Error (%)', fontsize=15)
ax_hist.set_ylabel('Density', fontsize=15)
ax_hist.spines['top'].set_visible(False)
ax_hist.spines['right'].set_visible(False)
ax_hist.legend()

bplot = ax_box.boxplot(ape.T, vert=False, autorange=False, widths=0.5, patch_artist=True, showfliers=False, boxprops=dict(facecolor='tomato'), medianprops=dict(color='black'))
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax_box.set(yticks=[])
ax_box.spines['left'].set_visible(False)
ax_box.spines['right'].set_visible(False)
ax_box.spines['top'].set_visible(False)

fig.suptitle('APE of the Posterior', fontsize=20)
plt.xlim(0, 100)
fig.tight_layout()
plt.savefig(file_path + '/data/ape_posterior2.png')
plt.clf()


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
plt.savefig(file_path + '/plots/sbc_rank_plot.png')
plt.clf()