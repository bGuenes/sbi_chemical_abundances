from Chempy.parameter import ModelParameters
from Chempy.cem_function import single_timestep_chempy

import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import sbi.utils as utils

import multiprocessing as mp
from tqdm import tqdm
import time


# ----- Set-Up -----------------------------------------------------------------------------------------------------------------------------------------------------
# --- Config ---
name = "TNG_train_data" # name of the data file
N_samples = int(5e4) # number of elements in test set

# --- Define the yield tables ---
yield_table_name_sn2_list = ['chieffi04','Nugrid','Nomoto2013','Portinari_net', 'chieffi04_net', 'Nomoto2013_net','NuGrid_net','West17_net','TNG_net','CL18_net']#'Frischknecht16_net'
yield_table_name_sn2_index = 8#5#8# use TNG 8#4#8#9
yield_table_name_sn2 = yield_table_name_sn2_list[yield_table_name_sn2_index]

yield_table_name_agb_list = ['Karakas','Nugrid','Karakas_net_yield','Ventura_net','Karakas16_net','TNG_net'] # Karakas2016 needs much more calculational resources (order of magnitude) using 2010 net yields from Karakas are faster and only N is significantly underproduced
yield_table_name_agb_index = 5#4#5 # use TNG #5#4#5#4
yield_table_name_agb = yield_table_name_agb_list[yield_table_name_agb_index]

yield_table_name_1a_list = ['Iwamoto','Thielemann','Seitenzahl', 'TNG']
yield_table_name_1a_index = 3#1#3 # use TNG #3#2#3#2
yield_table_name_1a = yield_table_name_1a_list[yield_table_name_1a_index]


# --- Define the prior ---
a = ModelParameters()
a.yield_table_name_sn2 = yield_table_name_sn2
a.yield_table_name_agb = yield_table_name_agb
a.yield_table_name_1a = yield_table_name_1a

# parameter labels
labels = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']
# parameter priors
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])
combined_priors = utils.MultipleIndependent(
    [Normal(p[0]*torch.ones(1), p[1]*torch.ones(1)) for p in priors] +
    [Uniform(torch.tensor([1.0]), torch.tensor([13.8]))],
    validate_args=False)

elements = a.elements_to_trace

# ----- Create parameters theta -----------------------------------------------------------------------------------------------------------------------------------
thetas = combined_priors.sample((N_samples,))
thetas = thetas.numpy()

# ----- Run Chempy ------------------------------------------------------------------------------------------------------------------------------------------------
def runner(index):
    """Function to compute the Chempy predictions for each parameter set"""
    #print(index)
    theta = thetas[index]
    a = ModelParameters()
    a.yield_table_name_sn2 = yield_table_name_sn2
    a.yield_table_name_agb = yield_table_name_agb
    a.yield_table_name_1a = yield_table_name_1a

    try:
        output=single_timestep_chempy((theta,a))
    except TypeError:
        output = np.inf
    if type(output)==float:
        if output==np.inf:
            del a
        outs=np.zeros(len(elements)),theta
    else: 
        abun=output[0]
        del a;
        outs=abun,theta
    return outs


if __name__ == '__main__':

    start = time.time()
    print("Running Chempy")
    with mp.Pool(mp.cpu_count()) as pool:
        output = list(tqdm(pool.imap(runner, np.arange(N_samples)), total=N_samples))
    abuns=[o[0] for o in output]
    thetas=[o[1] for o in output]
    end = time.time()

    print(f"Time taken: {end - start:.1f} s")
    np.savez(f'data/chempy_data/{name}.npz', params=thetas, abundances=abuns, elements=elements)