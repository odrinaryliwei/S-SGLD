# S-SGLD
the Sparse Stochastic Gradient Langevin Dynamics 

This project is the Sparse stochastic Gradient Langevi Dynamics Algorithm.  
We apply our method on MNIST(MNIST-FASHION) dataset.

Output of our code consists 12 columns. Columns 1-10  are the <img src="https://latex.codecogs.com/gif.latex?$$\mathcal{R}^2$$ />  with respect to each class of testing data. column 11 is the overall predictive accuracy of testing data, and column 12 is the sparsity of parameters.

The initial parameters, as specified in the main paper, are stored in `opt`.

We also include parameter ``

