# Inferring Galactic Parameters from Chemical Abundances with Simulation-Based Inference
$CHEMPY$ is a chemical evolution code that simulates the chemical evolution of galaxies. <br>
We use $CHEMPY$ to simulate chemical abundances of stars in a galaxy. <br>
We then train a neural network (NN) to learn the mapping from galactic parameters to chemical abundances to replace $CHEMPY$ as simulator and create more training points for the Neural Posterior Estimator (NPE). <br>
Finally we use the NN to train a NPE to infer the galactic parameters from the chemical abundances.

Our goal is to infer the global galactic parameters (initial mass function high-mass slope $\alpha_{IMF}$ & frequency of type Ia supernovae $log_{10}N_{Ia}$) from the chemical abundances of stars in a galaxy. <br>

The data used to train the NN and NPE is created with $CHEMPY$ as simulator and the TNG yield set in [``` chempy_test_data.py ```](chempy_test_data.py). <br>

<p align="center">
  <img src="plots/sbi_overview.png" />
</p>

## 1. Train NN
Firstly we train a neural network to learn the mapping from chemical abundances to galactic parameters. For this we use data created with $CHEMPY$ as simulator. <br>
The NN is trained on $\sim 500,000$ data points and validated on $\sim 50,000$ data points. The batch size is set to $64$ and the learning rate is set to $0.001$ and trained for $20$ epochs. <br>
The NN is a simple feed-forward neural network with $2$ hidden layers and $100$ neurons in the first and $40$ neurons in the second layer. <br>
That is sufficient for the accuracy of the generated data, since its absolute percantage error (APE) of $1.6^{+2.4}_{-0.9}\\%$ on the validation set is far below the error rate of real world data of $5\\%$. <br>
It took around $50s$ to train the NN on CPU. <br>

<div style="display: flex; justify-content: space-between;">
  <img src="plots/loss_NN_simulator.png" style="width: 49%;"/>
  <img src="plots/ape_NN.png" style="width: 49%;"/>
</div>

## 2. Train SBI
Secondly we use the NN to train a Neural Posterior Estimator (NPE). <br>
For that a total of $10^5$ datapoints simulated with the NN are used to train the NPE until it converges.
This takes approximatley $11$ minutes on CPU. <br>
The accuracy is afterwards tested with the $\sim 50,000$ validation data points from the original simulator $CHEMPY$. Each observation is sampled $1000$ times and the mean is compared to the ground truth. <br>
The NPE is has an absolute percantage error (APE) of $9.1^{+16.6}_{-6.2}\\%$ for a single prediction. <br>

<div style="display: flex; justify-content: space-between;">
  <img src="plots/sbc_rank_plot_1e5.png" style="width: 49%;"/>
  <img src="plots/ape_posterior2_1e5.png" style="width: 49%;"/>
</div>

The accuracy for a single prediction of the parameters is not really high. That's why we use multiple stars from the same galaxy to infer the global galactic parameters $\alpha_{IMF}$ & $log_{10}N_{Ia}$, since they are the same for all stars in the same galaxy. <br>

## 3. Inference

Finally we sample from the posterior distribution to infer the global galactic parameters. <br>
For that we use the chemical abundances of $1000$ stars from the same galaxy (meaning they have the same galactic parameters $\Lambda$) and the local parameters $\Theta_i$ are sampled from the priors.
The galactic parameters were fixed to  $\alpha_{IMF} = -2.3$ and $log_{10}N_{Ia} = -2.89$. <br>
The first set is created with the NN trained on data created with the TNG yield set. <br>
The second set is created from $CHEMPY$ directly with an alternative yield set. <br>
The third set is data created with the TNG simulator. <br>

### TNG yield set
<p align="center">
  <img src="plots/sbi_Nstar_comp.png" />
</p>

### Alternative yield set
<p align="center">
  <img src="plots/sbi_Nstar_analysis_alt.png" />
</p>

### TNG simulation data
<p align="center">
  <img src="plots/sbi_Nstar_analysis_tng.png" />
</p>

<div style="display: flex; justify-content: space-between;">
  <img src="plots/sbi_1000stars_noise.png" style="width: 33%;"/>
  <img src="plots/sbi_1000stars_noise_alt.png" style="width: 33%;"/>
  <img src="plots/sbi_1000stars_noise_tng.png" style="width: 33%;"/>
</div>

| | $CHEMPY$ TNG yield set | $CHEMPY$ Alternative yield set | TNG simulation data |
|---|---|---|---|
| SN Ia | TNG_net | Thielemann et al. (2003) |
| SN II | TNG_net | Nomoto et al. (2013) |
| AGB | TNG | Karakas & Lugaro (2016) |
| $\alpha_{IMF}$ | $-2.294 \pm 0.003$ | $-2.385 \pm 0.003$ | $-2.270 \pm 0.005$ |
|$\log_{10}N_{Ia}$| $-2.888 \pm 0.005$ | $-3.072 \pm 0.007$ | $-2.913 \pm 0.006$ |
| $\Delta\alpha_{IMF}$ | $0.26\\% $ | $3.7\\%$ | $1.3\\%$ |
| $\Delta\log_{10}N_{Ia}$ | $0.07\\%$ | $6.3\\%$ | $0.8\\%$ |

As expected, the inferred parameters deviate from the ground truth for a sigle prediction, since the NPE has a high error rate, 
but is able to infer the global parameters with a high accuracy for a growing number of stars in the case where we used data created with the correct yield set
that the posterior was trained on. 
The prediction for the TNG simulator seems also to be quite close to the ground truth. <br>
The deviation is higher for the alternative yield set, since the NN was trained on the TNG yield set and the NPE is not able to generalize to other yield sets. <br>
The total inference time for $1000$ simulations for the $1000$ stars is around $1$ minute for each yield set and therefore in a reasonable range compared to traditional MCMC methods. <br>