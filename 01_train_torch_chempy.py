import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Chempy.parameter import ModelParameters

import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import time as t
import os

# ----- Load the data ---------------------------------------------------------------------------------------------------------------------------------------------
# --- Load in training data ---
path_training = os.path.dirname(__file__) + '/data/chempy_data/TNG_Training_Data.npz'
training_data = np.load(path_training, mmap_mode='r')

elements = training_data['elements']
train_x = training_data['params']
train_y = training_data['abundances']


# ---  Load in the validation data ---
path_test = os.path.dirname(__file__) + '/data/chempy_data/TNG_Test_Data.npz'
val_data = np.load(path_test, mmap_mode='r')

val_x = val_data['params']
val_y = val_data['abundances']


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
val_x, val_y     = clean_data(val_x, val_y)

# convert to torch tensors
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
val_x = torch.tensor(val_x, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32)


# ----- Define the prior ------------------------------------------------------------------------------------------------------------------------------------------
a = ModelParameters()
labels = [a.to_optimize[i] for i in range(len(a.to_optimize))] + ['time']
priors = torch.tensor([[a.priors[opt][0], a.priors[opt][1]] for opt in a.to_optimize])


# ----- Define the model ------------------------------------------------------------------------------------------------------------------------------------------

class Model_Torch(torch.nn.Module):
    def __init__(self):
        super(Model_Torch, self).__init__()
        self.l1 = torch.nn.Linear(train_x.shape[1], 100)
        self.l2 = torch.nn.Linear(100, 40)
        self.l3 = torch.nn.Linear(40, train_y.shape[1])

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x

model = Model_Torch()
model


# ----- Train the model -------------------------------------------------------------------------------------------------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# shuffle the data
index = np.arange(train_x.shape[0])
np.random.shuffle(index)
train_x = train_x[index]
train_y = train_y[index]

# --- Train the neural network ---
epochs = 20
batch_size = 64
ep_loss = []
start = t.time()
for epoch in range(epochs):
    start_epoch = t.time()
    train_loss = []
    for i in range(0, train_x.shape[0], batch_size):
        optimizer.zero_grad()
        
        # Get the batch
        x_batch = train_x[i:i+batch_size].requires_grad_(True)
        y_batch = train_y[i:i+batch_size].requires_grad_(True)
        
        # Forward pass
        y_pred = model(x_batch)

        # Compute Loss
        loss = loss_fn(y_pred, y_batch)
        train_loss.append(loss.item())
        
        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()
        
    # Validation loss
    y_pred = model(val_x)
    val_loss = loss_fn(y_pred, val_y)
    
    train_loss = np.array(train_loss).mean()
    ep_loss.append([train_loss, val_loss.item()])
        
    end_epoch = t.time()
    epoch_time = end_epoch - start_epoch
    
    print(f'Epoch {epoch+1}/{epochs} in {round(epoch_time,1)}s, Loss: {round(train_loss,6)} | Val Loss: {round(val_loss.item(),6)}')
print(f'Training finished | Total time: {round(end_epoch - start, 1)}s')


# ----- Plot the loss -----
ep_loss = np.array(ep_loss)

plt.plot(np.arange(epochs)+1, ep_loss[:,0], label='Training Loss')
plt.plot(np.arange(epochs)+1, ep_loss[:,1], label='Validation Loss')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('MSE Loss', fontsize=14)
plt.title('Training and Validation Loss', fontsize=14)
plt.legend()
plt.savefig("plots/loss_NN_simulator.png")
plt.clf()

# ----- Calculate the L1 error -----

l1_err = torch.abs(model(val_x) - val_y).detach().cpu().numpy()
p1,p2,p3=np.percentile(l1_err,[15.865,50.,100-17.865],axis=0).mean(axis=1)

plt.hist(l1_err.flatten(), range=[0,.08], bins=100, density=True)
plt.xlabel(r'L1 Error [dex]', fontsize=14)
plt.ylabel(r'PDF', fontsize=14)

plt.title(r'L1 error (averaged across elements): %.3f-%.3f+%.3f'%(p2,p2-p1,p3-p2), fontsize=14)
plt.savefig("plots/l1_error_NN_simulator.png")
plt.clf()

# ----- Save the model --------------------------------------------------------------------------------------------------------------------------------------------
torch.save(model.state_dict(), 'data/pytorch_state_dict.pt')
print("Model trained and saved")
