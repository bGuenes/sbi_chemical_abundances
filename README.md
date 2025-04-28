# Inferring Galactic Parameters from Chemical Abundances with Simulation-Based Inference

[![arXiv](https://img.shields.io/badge/arXiv-2503.02456-b31b1b.svg)](https://arxiv.org/abs/2503.02456)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.14925307.svg)](https://doi.org/10.5281/zenodo.14925307)

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
Firstly we train a neural network to learn the mapping from chemical abundances to galactic parameters. For this we use data created with $CHEMPY$ as simulator over a uniform prior over the $5\sigma$-range of the original gaussian prior. <br>
The NN is trained on $\sim 100,000$ data points and validated on $\sim 50,000$ data points. The batch size is set to $64$ and the learning rate is set to $0.001$ and trained for $20$ epochs. We use [schedualfree](https://arxiv.org/abs/2405.15682) as optimizer. <br>
The NN is a simple feed-forward neural network with $2$ hidden layers and $100$ neurons in the first and $40$ neurons in the second layer. <br>
That is sufficient for the accuracy of the generated data, since its absolute percantage error (APE) of $2.1^{+3.0}_{-1.2}\\%$ on the validation set is far below the error rate of real world data of $5\\%$. <br>
It took around $200s$ to train the NN on CPU. <br>

|||
:-------------------------:|:-------------------------:
![](plots/loss_NN_simulator_5sigma_uni_prior.png)  |  ![](plots/ape_NN_5sigma_uni_prior.png)


## 2. Train NPE
Secondly we use the NN to train a Neural Posterior Estimator (NPE). <br>
The network is a Masked Autoregressive Flow (MAF) with $8$ hidden features and $4$ transforms. <br>
For that a total of $10^5$ datapoints simulated with the NN are used to train the NPE until it converges.
The parameters are sampled from a uniform prior over the $5\sigma$-range of the original gaussian prior to provide a better coverage of the parameterspace. <br>
This takes approximatley $20$ minutes on multiple CPUs. <br>
The accuracy is afterwards tested with the $\sim 50,000$ validation data points from the original simulator $CHEMPY$. Each observation is sampled $1000$ times and the mean is compared to the ground truth. <br>
The NPE is has an absolute percantage error (APE) of $10.7_{-7.5}^{+20.3}\\%$ for a single prediction and around $2.5\\%$ for the global parameters $\Lambda$, which we are interested in.<br>
The accuracy for a single prediction of the parameters is not really high. That's why we use multiple stars from the same galaxy to infer the global galactic parameters $\alpha_{IMF}$ & $log_{10}N_{Ia}$, since they are the same for all stars in the same galaxy. <br>

![](plots/ili_coverage_NPE_C_maf_8_4.png)
![](plots/ili_histogram_NPE_C_maf_8_4.png)
![](plots/ili_predictions_NPE_C_maf_8_4.png)

## 3. Inference

Finally we sample from the posterior distribution to infer the global galactic parameters. <br>
For that we use the chemical abundances of $200$ stars from the same galaxy (meaning they have the same galactic parameters $\Lambda$) and the local parameters $\Theta_i$ are sampled from the priors.
The galactic parameters were fixed to  $\alpha_{IMF} = -2.3$ and $log_{10}N_{Ia} = -2.89$. <br>
The first set is created with the NN trained on data created with the TNG yield set. <br>
The second set is created from $CHEMPY$ directly with an alternative yield set. <br>
The third set is data created with the TNG simulator. <br>

## 4. Multistar Posterior
We can compute the posterior for a single star from the samples from the NPE. <br>
Because of the central limit theorem, the posterior for a single star is a multivariate Gaussian. We can fit this with the sampled parameters from the NPE. <br>
This gives us the mean and covariance of $\alpha_{IMF}$ and $log{N_{Ia}}$ for one observation. <br>

The factorization for the posterior distribution  $P(\Lambda|\mathbf{x})$ can be obtained by applying Bayes rule twice:

$$ 
\begin{align*}
P(\Lambda| data) &\propto P(\Lambda)P(data|\Lambda) \\
&= P(\Lambda) \prod_ {i=1}^{N_{stars}} P(obs_i|\Lambda) \\ \\
&\propto P(\Lambda) \prod_ {i=1}^{N_{stars}} \frac{P(\Lambda|obs_i)}{P(\Lambda)} \\ \\
&= P(\Lambda)^{1-N_{stars}} \prod_ {i=1}^{N_{stars}} P(\Lambda|obs_i)\\ \\
&= \exp \left(-\frac{(1-N_ {stars})(\Lambda-\mu_ {prior})^2}{2\sigma_ {prior}^2}\right)\prod_ {i=1}^{N_{stars}} \exp \left(-\frac{1}{2} \frac{(\Lambda-\mu_i)^2}{\sigma_i^2}\right)
\end{align*} 
$$

The product of the single star posteriors is a product of Gaussians, so it's also a Gaussian with mean $\mathbf{\mu'}$ and variance $\mathbf{\sigma'}$:

$$
\begin{align*}
\mathbf{\mu'} &= \frac{\sum_{i=1}^{N_{stars}} \frac{\mu_i}{\sigma_i^2}}{\sum_{i=1}^{N_{stars}} \frac1{\sigma_i^2}} \\ \\
\mathbf{\sigma'}^2 &= \frac1 {\sum_{i=1}^{N_{stars}} \frac1{\sigma_i^2}}
\end{align*}
$$

In our case the prior for the galactic parameters $\Lambda$ is a gaussian as well. Therefore the resulting factorized posterior is again a gaussian and can be expressed with mean $\mathbf{\mu}$ and variance $\mathbf{\sigma}$:

$$
\begin{align*}
\mathbf{\mu} &= \frac{\frac{\mu'}{\sigma'^2}-\frac{(1-N)\mu_ {prior}}{\sigma_ {prior}^2}}{\frac1{\sigma'^2}-\frac{(1-N)}{\sigma_ {prior}^2}} \\ \\
\mathbf{\sigma}^2 &= \frac1 {\frac1{\sigma'^2}-\frac{(1-N)}{\sigma_ {prior}^2}}
\end{align*}
$$

### $CHEMPY$ TNG yield set
<p align="center">
  <img src="plots/CHEMPY TNG yields N_star comp.png" />
</p>

### $CHEMPY$ Alternative yield set
<p align="center">
  <img src="plots/CHEMPY alternative yields N_star.png" />
</p>

### TNG simulation data
<p align="center">
  <img src="plots/TNG simulation N_star.png" />
</p>

| | $CHEMPY$ TNG yield set | $CHEMPY$ Alternative yield set | TNG simulation data |
---|---|---|---
||![](plots/CHEMPY%20TNG%20yields.png)  |  ![](plots/CHEMPY%20alternative%20yields.png) | ![](plots/TNG%20simulation.png)
| SN Ia | TNG_net | Thielemann et al. (2003) |
| SN II | TNG_net | Nomoto et al. (2013) |
| AGB | TNG | Karakas & Lugaro (2016) |

As expected, the inferred parameters deviate from the ground truth for a sigle prediction, since the NPE has a high error rate,
but is able to infer the global parameters with a high accuracy for a growing number of stars in the case where we used data created with the correct yield set that the posterior was trained on. 
The model is still able to predict the parameters for the TNG simulation data to a good degree, but is overconfident. <br>
The prediction for a different yield set is far off, as expected since the dynamics of the simulator changed. <br>
The total inference time for $1000$ simulations for the $200$ stars is around $20$ seconds for each yield set and therefore orders of magnitudes faster then traditional MCMC methods, which would take around $40$ hours for $200$ stars. <br>


## 6. Test different galactic parameters
We test the NPE by sampling from a different values for the galactic parameters. <br>

<p align="center">
  <img src="plots/CHEMPY TNG yields N_star test.png" />
</p>
