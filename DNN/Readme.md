One could find Minst-Fashion dataset from the following link:
https://github.com/zalandoresearch/fashion-mnist/tree/master/data

In matlab Version,
set options.gpu = 1 if apply the algorithm with GPU and 0 otherwise
file `processMNISTimages.m` and `processMNISTlabels.m` preprocess the data.

Both Matlab and Keras version use same setup described in the paper.

We initialize the sampler from the full model with all components active, and the parameter initialized using the default initialization in Matlab/keras. We then run SA-SGLD for Niter = 250,000 iterations and we use the first 150,000 as burn-in.

![image](https://github.com/odrinaryliwei/S-SGLD/blob/master/DNN/Acc-Sparse.png)

As we could observe from the Accuracy-Sparsity plot, the algorithm achieves high prediction accuracy 0.867(0.002), whereas fairly low sparsity 0.005(0.000).

