# SA-SGLD
the Asynchronous Sparse Stochastic Gradient Langevin Dynamics 

This project is the Asynchronous Sparse Stochastic Gradient Langevin Dynamics Algorithm.

We propose a fast MCMC method to sample from the posterior distribution with a spike and slab prior. We further apply sub-sampling and employ stochastic gradient Langevin dynamics to reduce computational cost, and propose Asynchronous Sparse Stochastic Gradient Langevin Dynamics Algorithm (SA-SGLD). We refer readers our paper https://arxiv.org/abs/2108.06446 for more detailed background information. 

We apply our method on linear regression and logistic regression model. File `linear_regression_asynchronous.m` compares the relative error, sensitivity, sparsity and running time of Exact/Asyn sampler.

File `logistic_regression_SASGLD.m` compares the relative error, sensitivity, sparsity and running time of Exact/Asynchronous/SA-SGLD and Variational Approximation algorithm. 

We also apply SA-SGLD algorithm with a deep neural network (Lenet-5 applied to MNIST-FASHION dataset). File `call_lenet5_SASGLD.m` and `algo_sasgld.m `provide an implementation of SA-SGLD with specific setup. We also provide a tensorflow version `new_lenet5_sasgld.py`.












