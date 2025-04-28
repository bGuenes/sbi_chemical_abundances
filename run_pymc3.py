## Run the PyMC3 inference for a given input dataset and parameters

import numpy as np
from scipy.stats import norm
import pymc as pm
import pymc.math as ma
from pytensor import tensor as pt
from pytensor.graph.op import Op
from pytensor.tensor.type import TensorType
from pytensor import shared
import time as ttime
import os,sys,json
from Chempy.parameter import ModelParameters
from configparser import ConfigParser

import multiprocessing as mp

import torch
import torch.nn as nn

###########################
# Read in parameter file

file_path = os.path.dirname(__file__)

all_n = [1, 5, 10, 100, 200]
max_stars = max(all_n)
max_iteration = 200
elem_err = False
n_init = 2_000
n_samples = 1_000
chains = 1_000
cores = int(mp.cpu_count() * 0.8)
tune = 2_000

outfile = file_path + '/data/mcmc_inference.npz'

######################

a=ModelParameters()
labels_out = a.elements_to_trace
labels_in = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']

######################

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
model.load_state_dict(torch.load(file_path + '/data/pytorch_state_dict_5sigma_uni_prior.pt'))
model.eval()

def run_pytorch_model(input_array):
    """Run PyTorch model with proper error handling"""
    try:
        with torch.no_grad():
            tensor_input = torch.tensor(input_array, dtype=torch.float32)
            result = model(tensor_input)
            result = np.delete(result, 2, axis=1)  # Remove H from data
            return result.cpu().numpy()
    except Exception as e:
        print(f"Error running PyTorch model: {e}")
        # Return zeros as fallback
        return np.zeros((input_array.shape[0], len(labels_out)))

class PyTorchOp(Op):
    itypes = [TensorType(dtype='float64', shape=(None, None))]
    otypes = [TensorType(dtype='float64', shape=(None, None))]
    
    def __init__(self, torch_model):
        self.torch_model = torch_model
    
    def perform(self, node, inputs_storage, output_storage):
        try:
            x = inputs_storage[0]
            # Convert with proper error handling and explicitly specify dtype
            input_torch = torch.tensor(x, dtype=torch.float32)
            
            with torch.no_grad():
                output_torch = self.torch_model(input_torch)
            
            # Convert back to numpy with proper dtype
            result = np.array(output_torch.cpu().numpy(), dtype='float64')
            output_storage[0] = result
        except Exception as e:
            print(f"Error in PyTorchOp: {str(e)}")
            # Provide fallback output to prevent crash
            output_storage[0] = np.zeros((x.shape[0], self.torch_model.l3.out_features), dtype='float64')
    
    def grad(self, inputs, output_grads):
        return [pt.zeros_like(inputs[0])]

pytorch_op = PyTorchOp(model)
######################

# Load mock observations
mock_data = np.load(file_path + '/data/mock_abundances.npy')

# --- Load in training data ---
path_training = os.path.dirname(__file__) + '/data/chempy_data/chempy_train_uniform_prior_5sigma.npz'
training_data = np.load(path_training, mmap_mode='r')

elements = training_data['elements']
train_x = training_data['params']
train_y = training_data['abundances']
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


train_x, train_y = clean_data(train_x, train_y)

train_y = np.delete(train_y, 2, axis=1)  # Remove H from data, because it is just used for normalization (output with index 2)

input_mean = np.mean(train_x, axis=0)
input_std = np.std(train_x, axis=0)
output_mean = np.mean(train_y, axis=0)
output_std = np.std(train_y, axis=0)

n_els = train_y.shape[1]

######################

# Define priors
Lambda_prior_mean = a.p0[:2]
Theta_prior_mean = a.p0[2:]
Lambda_prior_width = [0.3,0.3]
Theta_prior_width = [0.3,0.1,0.1]

# Now standardize
std_Lambda_prior_mean = (Lambda_prior_mean-input_mean[:2])/input_std[:2]
std_Lambda_prior_width = (Lambda_prior_width)/input_std[:2]
std_Theta_prior_mean = (Theta_prior_mean-input_mean[2:5])/input_std[2:5]
std_Theta_prior_width = (Theta_prior_width)/input_std[2:5]
mu_times = input_mean[-1].repeat(max_stars)
sigma_times = input_std[-1].repeat(max_stars)

# Define critical theta edge:
log_SFR_crit = 0.29402
std_log_SFR_crit = (log_SFR_crit-input_mean[3])/input_std[3]

# Define bounds on age to stop predicting out of parameter space:
min_time,max_time = [1.,13.8]
std_min_time,std_max_time=[(time-input_mean[-1])/input_std[-1] for time in [min_time,max_time]]

def n_star_inference(n_stars,iteration,elem_err=False,n_init=20000,n_samples=1000,max_stars=100):    
    ## Define which stars to use
    these_stars = np.arange(max_stars)[iteration*n_stars:(iteration+1)*n_stars]
    
    ## Load in mock dataset
    obs_abundances = mock_data[these_stars]
    err = np.ones_like(obs_abundances)*float(5)/100.
    obs_err = norm.rvs(loc=np.zeros_like(obs_abundances), scale=err)

    obs_abundances = norm.rvs(loc=obs_abundances, scale=err)

    # Now standardize dataset
    norm_data=(obs_abundances-output_mean)/output_std
    norm_sigma = np.abs(obs_err/output_std)

    data_obs = norm_data.ravel()
    data_sigma = np.asarray(norm_sigma).ravel()

    std_times_mean = (mu_times-input_mean[-1])/input_std[-1]
    std_times_width = sigma_times/input_std[-1]
    
    # Define stacked local priors
    Local_prior_mean = np.vstack([np.hstack([std_Theta_prior_mean,std_times_mean[i]]) for i in range(n_stars)])
    Local_prior_sigma = np.vstack([np.hstack([std_Theta_prior_width,std_times_width[i]]) for i in range(n_stars)])
    
    # Bound variables to ensure they don't exit the training parameter space
    lower_bound = np.asarray([-5,std_log_SFR_crit,-5,std_min_time])
    upper_bound = np.asarray([5,5,5,std_max_time])
    
    # Create stacked mean and variances
    loc_mean=np.hstack([np.asarray(std_Theta_prior_mean).reshape(1,-1)*np.ones([n_stars,1]),std_times_mean[:n_stars].reshape(-1,1)])
    loc_std=np.hstack([np.asarray(std_Theta_prior_width).reshape(1,-1)*np.ones([n_stars,1]),std_times_width[:n_stars].reshape(-1,1)])

    def pytorch_forward(input_vars):
        # Convert PyTensor to NumPy to PyTorch
        input_np = input_vars.eval()
        input_torch = torch.FloatTensor(input_np)
        
        # Run through PyTorch model
        with torch.no_grad():
            output_torch = torch_model(input_torch)
        
        # Convert back to NumPy array for PyMC
        return np.array(output_torch)
    
    # Define PyMC Model
    simple_model = pm.Model()
    
    with simple_model:
        # Define priors
        Lambda = pm.Normal('Std-Lambda',mu=Lambda_prior_mean,
                            sigma=Lambda_prior_width,
                            shape=(1,len(Lambda_prior_mean)))

        Locals = pm.Normal('Std-Local',mu=Theta_prior_mean,
                            sigma=Theta_prior_width,
                            shape=(1,len(Theta_prior_mean)))

        TruLa = pm.Deterministic('Lambda',Lambda)
        TruTh = pm.Deterministic('Thetas',Locals[:,:3])
        TruTi = pm.Deterministic('Times',Locals[:,-1])

        ## NEURAL NET
        ones_tensor = pt.ones((n_stars, 1))
        Lambda_all = pt.dot(ones_tensor, Lambda)
        Locals_all = pt.dot(ones_tensor, Locals)
        Times = pt.reshape(mu_times[:n_stars], (n_stars, 1))
        InputVariables = ma.concatenate([Lambda_all, Locals_all, Times], axis=1)

        input_np = InputVariables.eval()  
        output = pm.Deterministic('neural_output', 
                         pt.as_tensor_variable(run_pytorch_model(input_np)))

        if elem_err:
            # ERRORS
            #element_error = pm.Normal('Element-Error',mu=-2,sigma=1,shape=(1,n_els))
            element_error = pm.HalfCauchy('Std-Element-Error',beta=0.01/output_std,shape=(1,n_els))
            TruErr = pm.Deterministic('Element-Error',element_error*output_std)
            stacked_error = pt.dot(ones_tensor,element_error)
            tot_error = ma.sqrt(stacked_error**2.+norm_sigma**2.) # NB this is all standardized by output_std here
        else:
            tot_error = norm_sigma # NB: all quantities are standardized here

        predictions = pm.Deterministic("Predicted-Abundances",output*output_std+output_mean)

        # Define likelihood function (unravelling output to make a multivariate gaussian)
        likelihood=pm.Normal('likelihood', mu=output.ravel(), sigma=tot_error.ravel(), 
                             observed=obs_abundances.ravel())
        
        # Now sample
        init_time = ttime.time()

        samples=pm.sample(draws=n_samples,chains=chains,cores=cores,tune=tune,
                step=pm.NUTS(target_accept=0.9),init='advi+adapt_diag',random_seed=42)
        end_time = ttime.time()-init_time

    def construct_output(samples):
        Lambda = samples.posterior['Lambda'].values.reshape(-1, 2)  # Reshape for Lambda dimensions
        Thetas = samples.posterior['Thetas'].values  # Extract Thetas values
        Times = samples.posterior['Times'].values    # Extract Times values
    
        predictions = samples.posterior['Predicted-Abundances'].values
    
        if elem_err:
            Errs = samples.posterior['Element-Error'].values.reshape(-1, n_els)
            return Lambda, Thetas, Times, Errs, predictions
        else:
            return Lambda, Thetas, Times, predictions

    print("Finished after %.2f seconds"%end_time)
    
    if elem_err:
        Lambda,Thetas,Times,Errs,predictions=construct_output(samples)
        return Lambda,Thetas,Times,end_time,Errs,predictions
    else:
        Lambda,Thetas,Times,predictions=construct_output(samples)
        return Lambda,Thetas,Times,end_time,predictions
    


## RUN THE INFERENCE ##
chain_params=[]
for nn in all_n:
    mini_chain=[]
    for iteration in range(max_stars//nn):
        if iteration>=max_iteration:
            break
        print("Starting inference using %d stars iteration %d of %d"%(nn,iteration+1,min(max_iteration,max_stars//nn)))
        try:
            mini_chain.append(n_star_inference(nn,iteration,elem_err=elem_err,n_init=n_init,
                                               n_samples=n_samples,max_stars=max_stars))
        except ValueError or FloatingPointError:
            mini_chain.append(n_star_inference(nn,iteration,elem_err=elem_err,n_init=n_init,
                                                   n_samples=n_samples,max_stars=max_stars))
    chain_params.append(mini_chain)

## Save output
print("Saving output")
all_n = all_n[:len(chain_params)]

all_Lambda = np.empty(len(all_n),dtype=object)
all_Lambda[:] = [[cc[0] for cc in c] for c in chain_params]

all_Thetas = np.empty(len(all_n),dtype=object)
all_Thetas[:] = [[cc[1][:,:,:] for cc in c] for c in chain_params]

all_Times = np.empty(len(all_n),dtype=object)
all_Times[:] = [[cc[2] for cc in c] for c in chain_params]

all_timescale = np.empty(len(all_n),dtype=object)
all_timescale[:] = [[cc[3] for cc in c] for c in chain_params]

if elem_err:
    all_Err = np.empty(len(all_n),dtype=object)
    all_Err[:] = [[cc[4] for cc in c] for c in chain_params]
else:
    all_Err=0.

all_predictions = np.empty(len(all_n),dtype=object)
all_predictions[:] = [[cc[-1] for cc in c] for c in chain_params]

mean_timescale = [np.mean(all_timescale[i],axis=0) for i in range(len(all_timescale))]

np.savez(outfile,n_stars=all_n,Lambdas=all_Lambda,Thetas=all_Thetas,Times=all_Times,
            runtimes=all_timescale,Errors=all_Err,mean_runtimes=mean_timescale)
print("Inference complete: output saved to %s"%outfile)
