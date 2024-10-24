# Inferring Galactic Parameters from Chemical Abundances with Simulation-Based Inference
$CHEMPY$ is a chemical evolution code that simulates the chemical evolution of galaxies. <br>
We use $CHEMPY$ to simulate chemical abundances of stars in a galaxy. <br>
We then train a neural network (NN) to learn the mapping from galactic parameters to chemical abundances to replace $CHEMPY$ as simulator and create more training points for the Neural Posterior Estimator (NPE). <br>
Finally we use the NN to train a NPE to infer the galactic parameters from the chemical abundances.

Our goal is to infer the global galactic parameters (initial mass function high-mass slope $\alpha_{IMF}$ & frequency of type Ia supernovae $log_{10}N_{Ia}$) from the chemical abundances of stars in a galaxy. <br>

<p align="center">
  <img src="plots/sbi3.png" />
</p>

## 1. Train NN
Firstly we train a neural network to learn the mapping from chemical abundances to galactic parameters. For this we use data created with $CHEMPY$ as simulator. <br>
The NN is trained on $\sim 500,000$ data points and validated on $\sim 50,000$ data points. The batch size is set to $64$ and the learning rate is set to $0.001$ and trained for $20$ epochs. <br>
The NN is a simple feed-forward neural network with $2$ hidden layers and $100$ neurons in the first and $40$ neurons in the second layer. <br>
That is sufficient for the accuracy of the generated data, since its absolute percantage error (APE) of $1.6^{+2.4}_{-0.9}\%$ on the validation set is far below the error rate of real world data of $5\%$. <br>
It took around $50s$ to train the NN on CPU. <br>

<div style="display: flex; justify-content: space-between;">
  <img src="plots/loss_NN_simulator.png" style="width: 49%;"/>
  <img src="plots/ape_NN.png" style="width: 49%;"/>
</div>

## 2. Train SBI
Secondly we use the NN to train a Neural Posterior Estimator (NPE). <br>
For that a total of $10^6$ datapoints simulated with the NN are used to train the NPE until it converges.
This takes approximatley $3$ h on CPU. <br>
The accuracy is afterwards tested with the $\sim 50,000$ validation data points from the original simulator $CHEMPY$. Each observation is sampled $1000$ times and the mean is compared to the ground truth. <br>
The NPE is has an absolute percantage error (APE) of $13.7^{+22.1}_{-10.6}\%$ for a single prediction. <br>

<div style="display: flex; justify-content: space-between;">
  <img src="plots/sbc_rank_plot.png" style="width: 49%;"/>
  <img src="plots/ape_posterior2.png" style="width: 49%;"/>
</div>

The accuracy for a single prediction of the parameters is not really high. That's why we use multiple stars from the same galaxy to infer the global galactic parameters $\alpha_{IMF}$ & $log_{10}N_{Ia}$, since they are the same for all stars in the same galaxy. <br>

## 3. Sample from Posterior
Finally we sample from the posterior distribution to infer the global galactic parameters. <br>
For that we use the chemical abundances of $1000$ stars from the same galaxy created with the NN. We used $\alpha_{IMF} = -2.3$ and $log_{10}N_{Ia} = -2.89$ and draw the local parameters from the prior distributions. <br>
The NPE is then used to infer the global galactic parameters. <br>
As expected, the inferred parameters deviate from the ground truth, since the NPE has a high error rate for a single prediction, but is able to infer the global parameters with a high accuracy for a growing number of stars. <br>
The total inference time for the $1000$ stars is around $5$ minutes.

<p align="center">
  <img src="plots/sbi_Nstar_comp.png" />
</p>

We can see that the estimated parameters $\alpha_{IMF}$ & $log_{10}N_{Ia}$ can be predicted with a high accuracy in a reasonable time compared to traditional MCMC methods. <br>

<p align="center">
  <img src="plots/sbi_1000stars_noise.png" style="width: 50%" />
</p>
