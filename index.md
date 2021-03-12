# Project 2

When we were finding the line of best fit for **univariate models** using **ordinary least squares (OLS)**, we were minimizing the sum of squared errors between the model and the data. 

The idea of minimizing some quantity is common to many model fitting and optimization problems, and the thing we are minimizing is referred as a *cost function*.

For **OLS**, we are minimizing the sum of the squared residuals in order to determine the slope and the intercept with the cost function:

![](https://latex.codecogs.com/svg.latex?\sum_{i=1}^N(y_i-\hat{y}_i)^2)

However, if the linear model contains many predictor variables or if these variables are correlated, the standard OLS parameter estimates have large variance, thus making the model unreliable. In other words, when we are working with multiple input variables for a **multivariate regression model**, we run into cases where the model becomes too complex by trying too hard to capture the noise in the training dataset. This will create an *overfit* model, which will probably yield poor prediction and generalization power. 

To mitigate this issue, we can **penalize the loss function** above by adding a multiple of an *L1* (Lasso) or an *L2* (Ridge) norm of the weights vector *w*(vector of the learned parameters in the linear regression). You get the following equation:


![](https://latex.codecogs.com/svg.latex?\L(X,Y)+\lambda{N(w)})

where the loss metric is *L(X,Y)*, and *N* is either the *L1*, *L2* or any other norm. 

Applying these penalization techniques is called **regularization**! Thus, the goal of regularization is to **reduce the complexity of the model** and possibly **prevent (or reduce) overfitting** by imposing some extra restrictions or conditions that will help us select one sparsity pattern or get the weights for all features. 

Another reason we might use regularization is if we have strong correlations in the data. We can visualize the correlations with a heap map.

This project will be demonstrating **different types of regularization techniques** on real-world datasets as well as simulated synthetic data with prescribed covariance to see how close each model can get to the expected outcome.


## Boston Housing Data

Let's First look at a heat map of the correlationa in the data: 
