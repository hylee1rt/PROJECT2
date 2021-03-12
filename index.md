# Project 2

When we were finding the line of best fit for **univariate models** using **ordinary least squares (OLS)**, we were minimizing the sum of squared errors between the model and the data. 

The idea of minimizing some quantity is common to many model fitting and optimization problems, and the thing we are minimizing is referred as a *cost function*.

For **OLS**, we are minimizing the sum of the squared residuals in order to determine the slope and the intercept with the cost function:

![\sum{i=1}^N(y_i-\hat{y_i})^2](https://render.githubusercontent.com/render/math?math=%5CLARGE+%5Cdisplaystyle+%5Csum_%7Bi%3D1%7D%5EN%28y_i-%5Chat%7By%7D_i%29%5E2%0A)

However, if the linear model contains many predictor variables or if these variables are correlated, the standard OLS parameter estimates have large variance, thus making the model unreliable. In other words, when we are working with multiple input variables for a **multivariate regression model**, we run into cases where the model becomes too complex by trying too hard to capture the noise in the training dataset. This will create an *overfit* model, which will probably yield poor prediction and generalization power. 

To mitigate this issue, we can **penalize the loss function** above by adding a multiple of an *L1* (Lasso) or an *L2* (Ridge) norm of the weights vector *w*(vector of the learned parameters in the linear regression). You get the following equation:

![\L(X,Y)+\lambda{N(w)}](https://render.githubusercontent.com/render/math?math=%5CLARGE+%5Cdisplaystyle+%5CL%28X%2CY%29%2B%5Clambda%7BN%28w%29%7D%0A)


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

We can compare the mean absolute errors that we obtained from the models introduced above: We can also plot a range of *a* to visualize where it may give us a minimum MAE.

| Model                          | MAE       | MAE (Standardized) |
|--------------------------------|-----------|--------------------|
| Linear Model                   | $3,640.02 |                    |                     
| MAE Ridge Regression Model     | $3,600.77 | $3,443.23          |    
| MAE Lasso Model                | $3,619.90 | $3,489.26          |       
| MAE Elastic Net Model          | $3,610.42 |  $3,452.95         | 


|Model                        | Optimal Alpha Value   |               
|--------------------------------|--------------------|
|<img src="https://user-images.githubusercontent.com/66886936/110965712-73365780-8322-11eb-8284-8da2b618fb16.png" width="500" height="400"  />|43.000|      
| <img src="https://user-images.githubusercontent.com/66886936/110965913-b42e6c00-8322-11eb-995f-ded5f5383fd6.png" width="500" height="400"  /> | 0.130 | 
| <img src="https://user-images.githubusercontent.com/66886936/110968953-205e9f00-8326-11eb-9e01-f83342c7c562.png" width="500" height="400"  />  | 0.130 | 


# SCAD - Fan & Li 2001

The **smoothly clipped absolute deviation (SCAD) penalty** was designed to encourage sparse solutions to the least squares problem, while also allowing for large values of *Beta*.

The cost function looks like:

![download (7)](https://user-images.githubusercontent.com/66886936/110971819-55202580-8329-11eb-94f8-10259f542246.png)

the SCAD penalty is often defined primarily by its first derivative 
p'(β), rather than p(β). Its derivative is


![p'_\lambda(\beta) = \lambda \left\{ I(\beta \leq \lambda) + \frac{(a\lambda - \beta)_+}{(a - 1) \lambda} I(\beta > \lambda) \right\}
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+p%27_%5Clambda%28%5Cbeta%29+%3D+%5Clambda+%5Cleft%5C%7B+I%28%5Cbeta+%5Cleq+%5Clambda%29+%2B+%5Cfrac%7B%28a%5Clambda+-+%5Cbeta%29_%2B%7D%7B%28a+-+1%29+%5Clambda%7D+I%28%5Cbeta+%3E+%5Clambda%29+%5Cright%5C%7D%0A)

where *a* is a tunable parameter that controls how quickly the penalty drops off for large values of β.

The penalty is defined as:

![\begin{cases} \lambda & \text{if } |\beta| \leq \lambda \\ \frac{(a\lambda - \beta)}{(a - 1) } & \text{if } \lambda < |\beta| \leq a \lambda \\ 0 & \text{if } |\beta| > a \lambda \\ \end{cases}
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Bcases%7D+%5Clambda+%26+%5Ctext%7Bif+%7D+%7C%5Cbeta%7C+%5Cleq+%5Clambda+%5C%5C+%5Cfrac%7B%28a%5Clambda+-+%5Cbeta%29%7D%7B%28a+-+1%29+%7D+%26+%5Ctext%7Bif+%7D+%5Clambda+%3C+%7C%5Cbeta%7C+%5Cleq+a+%5Clambda+%5C%5C+0+%26+%5Ctext%7Bif+%7D+%7C%5Cbeta%7C+%3E+a+%5Clambda+%5C%5C+%5Cend%7Bcases%7D%0A)

and as implemented as follows.

```python
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
    
    def scad(beta):
  beta = beta.flatten()
  beta = beta.reshape(-1,1)
  n = len(y)
  return 1/n*np.sum((y-X.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))
  
def dscad(beta):
  beta = beta.flatten()
  beta = beta.reshape(-1,1)
  n = len(y)
  return np.array(-2/n*np.transpose(X).dot(y-X.dot(beta))+scad_derivative(beta,lam,a)).flatten()
```


# Square Root Lasso

**Square root lasso** is a modification of the Lasso. Belloni et al.  proposed a pivotal method for estimating high-dimensional sparse linear regression models. For this model, we do not need to know the standard deviation of the noise. 

The cost function is:

![\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}+\alpha\sum_{i=1}^{p}|\beta_i|
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y_i-%5Chat%7By%7D_i%29%5E2%7D%2B%5Calpha%5Csum_%7Bi%3D1%7D%5E%7Bp%7D%7C%5Cbeta_i%7C%0A)

and the implementation of the function will return us *yhat*, the predictions, and *beta*, which are the coefficients.

```python 
def sqrtLasso(X,y,alpha):
    model = sm.OLS(y,X)
    result = model.fit_regularized(method='sqrt_lasso', alpha=alpha,profile_scale=True)
    return result.predict(X), result.params

yhat, beta = sqrtLasso(X,y,0.5)
```

| Model                          | MAE      | 
|--------------------------------|----------|
| MAE SCAD Model                 |$3,230.29 |                    
| MAE Square Root Lasso Model    |$3,258.48 |                    



So up to now, we have been working with a single random split of the data. What if we just happened to pick a particular random_state which gave us unusually high or low results, and only did this test once?  We could possibly be misled about how good or bad our model is (what if you flipped a coin 10 times and got 10 heads...) Maybe try 100 more times to check?

To resolve some of these concerns, we will briefly take a look at a concept called **K-fold cross validation**, and here's how it goes:

1. Split your data into *K* equally sized groups (you pick this number).  These groups are called **folds**.
2. Use the first fold as your test data, and the remining *K-1* folds as your training data, and then check the scores.
4. Use the second fold as your test data, and the remaining *K-1* folds as your training data.
5. Repeat this process *K* times, using each of the *K* folds as your test data exactly once.


This method of testing the model on different training and test sets can be implemented as:

```python
def DoKFold(X,y,alpha,n):
  mae = []
  kf = KFold(n_splits=n,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    model = sm.OLS(y_train,X_train)
    result = model.fit_regularized(method='sqrt_lasso', alpha=alpha,profile_scale=True)
    yhat_test = result.predict(X_test)
    mae.append(mean_absolute_error(y_test,yhat_test))
  return np.mean(mae)
```

The result of this K-fold validation will give us the average MAE of the folds at each *a*. This should give us a better sense of the errors, and provide greater confidence in the outcome. Now, what if we change around our hyperparameter *a*? How would our MAE change?

- Lowest MAE value:  $3,579.26 
- Optimal alpha value: 0.100|

With cross validation, the lowest MAE value from our cross validation square root lasso model looks more similar to the MAE from other models. 

<img src="https://user-images.githubusercontent.com/66886936/110978061-e8108e00-8330-11eb-9824-c980583d850a.png" width="600" height="400"  />


# Applying the models on simulated synthetic data

To better understand our models and thus our data, we will use a Toeplitz matrix implementation to simulate multiple correlations. Below, we will define a function to generate X with the given number of observations as num_samples, number of features as p, and the strength of the correlation, rho. It will return X with the Toeplitz correlation structure.

```python 
def make_correlated_features(num_samples,p,rho):
  vcor = [] 
  for i in range(p):
    vcor.append(rho**i)
  r = toeplitz(vcor)
  mu = np.repeat(0,p)
  X = np.random.multivariate_normal(mu, r, size=num_samples)
  return X
  
n = 200
p = 50
X = make_correlated_features(200,p,0.8)
  ```
  
  Simulate some **ground truth** for the data (this will be linear)
  
  ![\large y = X*\beta^* +\sigma\epsilon](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Clarge+y+%3D+X%2A%5Cbeta%5E%2A+%2B%5Csigma%5Cepsilon%0A)
  
  
  and fill in zeros for the rest of the coefficients.
  
  ```python
  beta = np.array([-1,2,3,0,0,0,0,2,-1,4])
beta = beta.reshape(-1,1)
betas = np.concatenate([beta,np.repeat(0,p-len(beta)).reshape(-1,1)],axis=0)
```
  
  Then we can generate *y* with some random noise which follows a normal distribution.
 
```python
n = 200
sigma = 2
y = X.dot(betas) + sigma*np.random.normal(0,1,n).reshape(-1,1)
```

Now that we have our simulated data, *X* and *y*, we could explore how the models perform on this dataset. We will take a single split of the data and fit the models:

```python
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3,random_state=1234)
```
  
  
  
| Model                          | MAE       | 
|--------------------------------|-----------|
| Linear Model                   | 2.03      |                                       
| MAE Ridge Regression Model     | 1.86      |                      
| MAE Lasso Model                | 1.85      |                         
| MAE Elastic Net Model          | 1.85      | 
  
  
  
  
  
  
