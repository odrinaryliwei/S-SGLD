# S-SGLD
the Sparse Stochastic Gradient Langevin Dynamics 

This project is the Sparse stochastic Gradient Langevi Dynamics Algorithm.  
We apply our method on MNIST(MNIST-FASHION) dataset.

file`algo_Lenet5MNIST.m` realizes Algorithm 2 in the main paper.

In file `calls_Lenet5MNIST.m`, 

`fashion` determines whether dataset is MNIST or MNIST-FASHION.

`Sigma` and `angle`(which are commented) are included when we add noise or make rotation to testing data.

The initial parameters, as specified in the main paper, are stored in `opt`.

Output `Res` consists 12 columns. Columns 1-10  are the overall <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{R}^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{R}^2" title="\mathcal{R}^2" /></a> with respect to each class of testing data. column 11 is the overall predictive accuracy of testing data, and column 12 is the sparsity of parameters.

Output `Rsq` stores <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{R}^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{R}^2" title="\mathcal{R}^2" /></a> w.r.t. each testing data.







