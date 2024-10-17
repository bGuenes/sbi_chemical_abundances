# Inferring Galactic Parameters from Chemical Abundances with Simulation-Based Inference

![](plots/sbi2.png)

## 1. Train NN
Firstly we train a neural network to learn the mapping from chemical abundances to galactic parameters. For this we use data created with $CHEMPY$ as simulator. <br>
The NN is trained on $\sim 500,000$ data points and validated on $\sim 50,000$ data points. The batch size is set to $64$ and the learning rate is set to $0.001$ and trained for $20$ epochs. <br>
The NN is a simple feedforward neural network with $2$ hidden layers and $100$ neurons in the first and $40$ neurons in the second layer. <br>
That is sufficient for the accuracy of the generated data, since its error rate of $0.006^{+0.006}_{-0.004}$ dex is far below the error rate of real world data of $0.05$ dex. <br>
It took around $50s$ to train the NN on CPU. <br>

<div style="display: flex; justify-content: space-between;">
  <img src="plots/loss_NN_simulator.png" style="width: 49%;"/>
  <img src="plots/l1_error_NN_simulator.png" style="width: 49%;"/>
</div>

## 2. Train SBI
Secondly we use the trained neural network to train a simulator-based inference algorithm. We use the Sequential Neural Posterior Estimation (SNPE) algorithm. <br>
For that a total of 1 million simulations are used to train the algorithm.

## 3. Sample from Posterior
Finally we sample from the posterior distribution to infer the galactic parameters.

## 4. Plot Results