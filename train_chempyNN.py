from Chempy.parameter import ModelParameters
from Chempy.cem_function import single_timestep_chempy
import numpy as np
from sklearn.neural_network import MLPRegressor


# --- Load & prepare the data -----------------------------------------------------------------------------------------

# --- Load in training data ---
path_training = '../ChempyMulti/tutorial_data/TNG_Training_Data.npz'
training_data = np.load(path_training, mmap_mode='r')

elements = training_data['elements']
train_x = training_data['params']
train_y = training_data['abundances']


# ---  Load in the validation data ---
path_test = '../ChempyMulti/tutorial_data/TNG_Test_Data.npz'
val_data = np.load(path_test, mmap_mode='r')

val_x = val_data['params']
val_y = val_data['abundances']


# --- Clean the data ---
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


# --- Normalize the data ---
x_mean, x_std = train_x.mean(axis=0), train_x.std(axis=0)
y_mean, y_std = train_y.mean(axis=0), train_y.std(axis=0)


def normalize_data(x, y, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std):
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std

    return x, y


train_x, train_y = normalize_data(train_x, train_y)
val_x, val_y     = normalize_data(val_x, val_y)


# add time squared as parameter
def add_time_squared(x):
    return np.concatenate((x, (x[:, -1]**2).reshape((len(x), 1))), axis=1)


train_x = add_time_squared(train_x)
val_x = add_time_squared(val_x)


# --- Train the neural network ----------------------------------------------------------------------------------------
# --- Define the neural network ---
def single_regressor(x, y, neurons=40, epochs=3000, verbose=False):
    """Return out-of-sample score for a given number of neurons for one element"""
    model = MLPRegressor(solver='adam', alpha=0.001, max_iter=epochs, learning_rate='adaptive', tol=1e-13,
                         hidden_layer_sizes=(neurons,), activation='tanh', verbose=verbose,
                         shuffle=True, early_stopping=True)

    model.fit(x, y)

    model_pred = model.predict(x)
    score = np.mean((model_pred-y)**2.)
    diff = np.abs(y-model_pred)

    w0, w1 = model.coefs_
    b0, b1 = model.intercepts_

    return score, diff, [w0, w1, b0, b1]


# --- Train the neural network ---
# Train an independent neural network for each element and save the weights
output = []
neurons = 40
for el_i, el in enumerate(elements):
    print("Running net %d of %d" % (el_i + 1, len(elements)))
    o = single_regressor(train_x, train_y[:, el_i], neurons=neurons, epochs=3000, verbose=False)
    print("Score for element %s is %.3f" % (el, o[0]))
    output.append(o)


# --- Save the neural network outputs ---
scores = [score for score, _, _ in output]
diffs = [diff for _, diff, _ in output]
coeffs = [co for _, _, co in output]

w0 = np.hstack([co[0] for co in coeffs])
b0 = np.hstack([co[2] for co in coeffs])
b1 = np.hstack([co[3] for co in coeffs])

# Read in w1 vector into sparse structure
w1 = np.zeros([w0.shape[1], b1.shape[0]])
assert neurons == w0.shape[1] / len(coeffs)
for i in range(len(coeffs)):
    w1[int(neurons * i):int(neurons * (i + 1)), i] = coeffs[i][1][:, 0]


# --- Save the weights and normalization parameters ---
# Save output
np.savez('data/tutorial_weights.npz',
         w0=w0, w1=w1, b0=b0, b1=b1,
         in_mean=x_mean, in_std=x_std, out_mean=y_mean, out_std=y_std,
         activation='tanh', neurons=neurons)
