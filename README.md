# SA-SGLD
the Asynchronous Sparse Stochastic Gradient Langevin Dynamics 

This project is the Asynchornous Sparse stochastic Gradient Langevi Dynamics Algorithm.

We propose a fast MCMC method to sample from the posterior distribution with a spike and slab prior. We further apply sub-sampling and employ stochastic gradient Langevin dynamics to reduce computational cost, and propose Asynchornous Sparse stochastic Gradient Langevi Dynamics Algorithm (SA-SGLD). We refer readers our paper `xxx` for more detailed background information. 

We apply our method on logistic regression model. File `live_lr_exact_asyn_sasgld_va.m` compares the relative error and running time of Exact/Asynchorous/SA-SGLD and Variational Approximation algorithm. 

We also apply SA-SGLD algorithm on with a deep neural network (Lenet-5 applied to MNIST-FASHION dataset).












