import numpy as np
from Chempy.parameter import ModelParameters
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


# --- Set-up the Network ----------------------------------------------------------------------------------------------
def stacked_net_output(in_par):
    l1 = np.matmul(in_par, w0) + b0
    return np.matmul(np.tanh(l1), w1) + b1


# --- Set-up priors ---------------------------------------------------------------------------------------------------
a = ModelParameters()
priors = np.array([[a.priors[opt][0],a.priors[opt][1]] for opt in a.to_optimize])


# --- create new data points ------------------------------------------------------------------------------------------
def add_time_squared(x):
    return np.concatenate((x, (x[:, -1]**2).reshape((len(x), 1))), axis=1)

samples = 1000000

# Random samples from the priors
data = np.random.normal(priors[:, 0], priors[:, 1], size=(samples, len(priors)))
# Random samples for the time
time = np.random.uniform(2, 12.8, samples).reshape(samples, 1)
data = np.concatenate((data, time), axis=1)

# Normalize the data
norm_data = (data - in_mean) / in_std

# Add time squared as parameter
norm_data = add_time_squared(norm_data)

# --- Predict the new data --------------------------------------------------------------------------------------------
start = t.time()
abundances = stacked_net_output(norm_data)
abundances = abundances * out_std + out_mean
print(f'Time taken for {samples} datapoints: {round(t.time() - start, 4)}s')


# --- Save the new data -----------------------------------------------------------------------------------------------
np.savez('data/training_data.npz', data=data, abundances=abundances)
