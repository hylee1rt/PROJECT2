# Project 2

When we were finding the line of best fit for **univariate models** using **ordinary least squares (OLS)**, we were minimizing the sum of squared errors between the model and the data. 

The idea of minimizing some quantity is common to many model fitting and optimization problems, and the thing we are minimizing is referred as a *cost function*.

For **OLS**, we are minimizing the sum of the squared residuals in order to determine the slope and the intercept with the cost function:

![\sum_{i=1}^N(y_i-\hat{y}_i)^2
](https://render.githubusercontent.com/render/math?math=%5CLARGE+%5Cdisplaystyle+%5Csum_%7Bi%3D1%7D%5EN%28y_i-%5Chat%7By%7D_i%29%5E2%0A)

However, if the linear model contains many predictor variables or if these variables are correlated, the standard OLS parameter estimates have large variance, thus making the model unreliable. In other words, when we are working with multiple input variables for a **multivariate regression model**, we run into cases where the model becomes too complex by trying too hard to capture the noise in the training dataset. This will create an *overfit* model, which will probably yield poor prediction and generalization power. 

To mitigate this issue, we can **penalize the loss function** above by adding a multiple of an *L1* (Lasso) or an *L2* (Ridge) norm of the weights vector *w*(vector of the learned parameters in the linear regression). You get the following equation:

![\L(X,Y)+\lambda{N(w)}
](https://render.githubusercontent.com/render/math?math=%5CLARGE+%5Cdisplaystyle+%5CL%28X%2CY%29%2B%5Clambda%7BN%28w%29%7D%0A)


where the loss metric is *L(X,Y)*, and *N* is either the *L1*, *L2* or any other norm. 

Applying these penalization techniques is called **regularization**! Thus, the goal of regularization is to **reduce the complexity of the model** and possibly **prevent (or reduce) overfitting** by imposing some extra restrictions or conditions that will help us select one sparsity pattern or get the weights for all features. 

Another reason we might use regularization is if we have strong correlations in the data. We can visualize the correlations with a heap map.

This project will be demonstrating **different types of regularization techniques** on real-world datasets as well as simulated synthetic data with prescribed covariance to see how close each model can get to the expected outcome.


## Boston Housing Data

Let's First look at a heat map of the correlationa in the data. The colors of the heat map will show us the strength of correlations between features and let us know if we should use regularization methods to account for those correlations better.  

![download](https://user-images.githubusercontent.com/66886936/110899861-d2b74780-82cf-11eb-97b6-159cbda0f83e.png)







# L2 (Ridge) Regularization - Tikhonov 1940's

The first regularization technique we will look at is commonly called **Ridge regression**.  In addition to minimizing the sum of squared errors, Ridge regression also penalizes a model for having more parameters and/or larger parameters.  This is accomplished by modifying the cost function:

![\sum_{i=1}^N(y_i-\hat{y}_i)^2+\alpha\sum_{i=1}^p\beta_i^2](https://render.githubusercontent.com/render/math?math=%5CLARGE+%5Cdisplaystyle+%5Csum_%7Bi%3D1%7D%5EN%28y_i-%5Chat%7By%7D_i%29%5E2%2B%5Calpha%5Csum_%7Bi%3D1%7D%5Ep%5Cbeta_i%5E2)


The new term in this equation, the one following *alpha*, is called the **regularization term**.  In words, it just says to square all of the model coefficients, add them together, multiply that sum by some number *alpha* and then add that to the cost function.  As a result, the model will be simultaneously trying to minimize both the sum of the squared errors as well as the number/magnitude of the model's parameters.

The *alpha* in the regularization term is referred to as a **hyperparameter** - which is a number that is not determined during the fitting/training procedure.  Different values of *a* will lead to different results, and for now, we will have to guess and find the optimal value ourselves.

The plot below to visualizes how the coefficients of the ridge regression model are changing as we adjust *a*.

![download (1)](https://user-images.githubusercontent.com/66886936/110906664-42323480-82da-11eb-8b51-8e6ad47eb43d.png)


# L1 (Lasso) Regularization - Tibshirani 1993

For **Lasso regression**, in stead of squaring the coefficients, we are taking their **absolute value**. Unlike Ridge, that is more reliable for predictive power, Lasso can set some of the model coefficients to zero, effectively removing variables from the model.

The cost function is defined as:

![\sum_{i=1}^N(y_i-\hat{y}_i)^2+\alpha\sum_{i=1}^p\lvert\beta_i\rvert](https://render.githubusercontent.com/render/math?math=%5CLARGE+%5Cdisplaystyle+%5Csum_%7Bi%3D1%7D%5EN%28y_i-%5Chat%7By%7D_i%29%5E2%2B%5Calpha%5Csum_%7Bi%3D1%7D%5Ep%5Clvert%5Cbeta_i%5Crvert)

The plot below to visualizes how the coefficients of the lasso regression model are changing as we adjust *a*. We see that the coefficients from lasso regression do not asymptotically reach near zero like ridge regression, but rather hit zero and disappear.

![download (2)](https://user-images.githubusercontent.com/66886936/110906859-93dabf00-82da-11eb-8e2f-de64cad57d75.png)


# Elastic Net

**Elastic net** is a penalized linear regression model that includes both the $L_1$ and $L_2$ penalties during training. The cost function is: 

![\hat{\beta} = argmin_\beta \left\Vert  y-X\beta \right\Vert ^2 + \lambda_2\left\Vert  \beta \right\Vert ^2 + \lambda_1\left\Vert  \beta\right\Vert_1
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Chat%7B%5Cbeta%7D+%3D+argmin_%5Cbeta+%5Cleft%5CVert++y-X%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_2%5Cleft%5CVert++%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_1%5Cleft%5CVert++%5Cbeta%5Cright%5CVert_1%0A)


for *a* in *[0,1]*. 

Using this model, you can find the combination that fits both multiple correlations in the data and the sparsity pattern. 
The plot below to shows how the coefficients of the Elastic Net model are changing as we adjust *alpha*. 

![download (3)](https://user-images.githubusercontent.com/66886936/110908053-48c1ab80-82dc-11eb-81c3-792438b28b36.png)




For now, let us compare the mean absolute errors that we obtained from the models introduced above.

#### Boston Housing Dataset
| Model                          | MAE       | MAE (Standardized) | Optimal Alpha Value |                
|--------------------------------|-----------|--------------------|---------------------|
| Linear Model                   | $3,640.02 |                    |                     |
| MAE Ridge Regression Model     | $3,600.77 | $3,443.23          |    43.000           |
| MAE Lasso Model                | $3,619.90 | $3,489.26          |       0.130         |
| MAE Elastic Net Model          | $3,610.42 |  $3,452.95         |       0.130         |


|Ridge                         | Lasso   | Elastic Net |                 
|--------------------------------|-----------|--------------------|
|<img src="https://user-images.githubusercontent.com/66886936/110965712-73365780-8322-11eb-8284-8da2b618fb16.png" width="400" height="400"  />|hello|  hello |         
| <img src="https://user-images.githubusercontent.com/66886936/110965913-b42e6c00-8322-11eb-995f-ded5f5383fd6.png" width="400" height="400"  />     | $3,600.77 | $3,443.23          |   
| <img src="ttps://user-images.githubusercontent.com/66886936/110967099-0328d100-8324-11eb-9c5d-406eacf90740.png" width="400" height="400"  />  | $3,619.90 | $3,489.26   |   
| MAE Elastic Net Model          | $3,610.42 |  $3,452.95         |       









