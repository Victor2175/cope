import torch     # type: ignore
import numpy as np

# 1- Ridge regression problem : 
# $\min_{W} \Vert Y - X W \Vert_F^2 + \lambda \Vert W \Vert_F^2$

def ridge_regression(X, Y, lambda_=1.0,dtype=torch.float32,verbose=False):
    """
    Computes the closed-form solution for reduced rank regression.
    
    Args:
        X (torch.Tensor): Predictor matrix of shape (n, p).
        Y (torch.Tensor): Response matrix of shape (n, q).
        lambda_ (scalar): Ridge penalty coefficient.
        
    Returns:
        U (torch.Tensor): Low-rank predictor coefficients of shape (p, rank).
        V (torch.Tensor): Low-rank response coefficients of shape (q, rank).
    """

    # compute Penroe Morose pseudo inverse of X^T @ X
    P = torch.linalg.inv(X.T @ X + lambda_ * torch.eye(X.shape[1],dtype=dtype))
    
    # compute ordinary least square solution 
    W_ols = P @ X.T @ Y

    # print loss function 
    if verbose:
        loss = torch.norm(Y - X @ W_ols,p='fro')**2 + lambda_ * torch.norm(W_ols,p='fro')**2
        print("Loss function: ", loss.item())
    
    return W_ols


# 2- Low-rank ridge regression problem: 
# $\min_{W \colon \mathrm{rank}(W) \leq r} \Vert Y - X W \Vert_F^2 + \lambda \Vert W \Vert_F^2$


def ridge_regression_low_rank(X, Y, rank=5.0, lambda_=1.0,dtype=torch.float32,verbose=False):
    """
    Computes the closed-form solution for reduced rank regression.
    
    Args:
        X (torch.Tensor): Predictor matrix of shape (n, p).
        Y (torch.Tensor): Response matrix of shape (n, q).
        rank (Int): Desired rank for the approximation.
        lambda_ (Float64): Ridge penalty coefficient.
        
    Returns:
        U (torch.Tensor): Low-rank predictor coefficients of shape (p, rank).
        V (torch.Tensor): Low-rank response coefficients of shape (q, rank).
    """

    # compute Penroe Morose pseudo inverse of X^T @ X
    P = torch.linalg.inv(X.T @ X + lambda_ * torch.eye(X.shape[1],dtype=dtype))
    
    # compute ordinary least square solution 
    W_ols = P @ X.T @ Y
    
    # compute SVD decomposition of X @ W_ols
    U, S, Vh = torch.linalg.svd(X @ W_ols, full_matrices=False)
    
    # Truncate to the desired rank
    U_r = U[:, :rank]            # (p, rank)
    S_r = torch.diag(S[:rank])   # (rank, rank)
    V_r = Vh[:rank, :].T         # (q, rank)

    # compute regressor
    W_rrr = W_ols @ V_r @ V_r.T

    # print loss function
    if verbose:
        loss = torch.norm(Y - X @ W_rrr,p='fro')**2 + lambda_ * torch.norm(W_rrr,p='fro')**2
        print("Loss function: ", loss.item())

    return W_rrr

# Function that returns the low-rank projection of a given matrix M using the Eckart–Young–Mirsky theorem.
# Proj_(rank <= r)(M) = U_r S_r V_r^T
# where U_r, S_r, V_r are the truncated SVD decomposition of M.  
def low_rank_projection(M,rank=5,dtype=torch.float32):
    """Compute low-rank projection of a given matrix M. Thanks to Eckart–Young–Mirsky theorem, we can derive a closed-form solution.

        Args:
            - M: torch.tensor (n x m)
            - rank: integer

        Returns:
            - M_low_rank: low-rank approaximation of matrix M.
    """

    # compute SVD decomposition of W
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    
    # Truncate to the desired rank
    U_r = U[:, :rank]            # (p, rank)
    S_r = torch.diag(S[:rank])   # (rank, rank)
    V_r = Vh[:rank, :].T         # (q, rank)

    # compute regressor
    M_low_rank = U_r @ S_r @ V_r.T 

    # assert that the rank is correct
    assert torch.linalg.matrix_rank(M_low_rank) == rank

    # return low-rank projection
    return M_low_rank

######## Functions used to solve the robust weight regression problem using accelerated gradient descent with low-rank projection #########
# 3 - $\max_{\alpha \in \Delta} \min_{W} \sum_{m} \alpha_m \Vert \Sigma^{-1/2}(Y_m - X_m W) \Vert_F^2 + \lambda \Vert W \Vert_F^2$

# compute gradient
def compute_gradient(models,x,y,w,notnan_idx,lambda_=1.0,mu_=1.0,dtype=torch.float32):
    """This function computes the gradient of ridge log-sum-exp loss with respect to W: 
    $\sum_{m,r} -(2/R^m) X_{m,r}.T (Y_{m,r} - X_{m,r} W) softmax(norm(Y_{m,r} - X_{m,r} W)) +  2\lambda * W$

    Args:
        - models: list of strings
        - x,y: dictionaries of input-output pairs per model and per run, and variances.
        - w: torch.tensor (grid_size, grid_size)
        - notnan_idx: list of integers
        - lambda_: torch.dtype, ridge penalty coefficient
        - mu_: torch.dtype, entropy penalty coefficient
        
    Returns:
        - Gradient matrix: torch.tensor grid_size x grid_size
    """
    res = torch.zeros(len(models), w.shape[0], w.shape[0], dtype=dtype)
    res_sumexp = torch.zeros(len(models), dtype=dtype)
    
    for idx_m, m in enumerate(models):

        # compute -2X_{m,r}^T (Y_{m,r}^T - X_{m,r}^T W)
        res[idx_m][np.ix_(notnan_idx,notnan_idx)] = - 2*torch.mean(torch.bmm(torch.transpose(x[m][:,:,notnan_idx], 1,2) , \
                                                        y[m][:,:,notnan_idx] - x[m][:,:,notnan_idx] @ w[np.ix_(notnan_idx,notnan_idx)]),dim=0, dtype=dtype)

        # compute the exponential term
        res_sumexp[idx_m] = (1/mu_)*torch.mean(torch.norm(y[m][:,:,notnan_idx] - x[m][:,:,notnan_idx] @ w[np.ix_(notnan_idx,notnan_idx)],p='fro',dim=(1,2))**2, dtype=dtype)
            
    res_sumexp = torch.nn.functional.softmax(res_sumexp,dim=0, dtype=dtype)

    # compute gradient as sum (res * softmax)
    grad = torch.sum(torch.unsqueeze(torch.unsqueeze(res_sumexp,-1),-1) * res, dim=0, dtype=dtype)
    grad[np.ix_(notnan_idx,notnan_idx)] = grad[np.ix_(notnan_idx,notnan_idx)] + 2*lambda_* w[np.ix_(notnan_idx,notnan_idx)]
    
    return grad 

def train_robust_weights(models,x,y,lon_size,lat_size,notnan_idx,\
                               rank=5.0,lambda_=1.0,mu_=1.0,\
                               lr=1e-5,nb_iterations=20,dtype=torch.float32,verbose=False):
    """This function runs accelerated gradient descent. If rank is not None, then it runs a low rank projection step at each iteration.

       Args:
        - models: list of strings, climate models taken into account
        - x,y: dictionaries of input-output pairs and variances per model 
        - lon_size, lat_size: integers, longitude-latitude grid size
        - notnan_idx: list of integers
        - rank: integer, low rank constraint
        - lambda_: torch.dtype, ridge penalty coefficient
        - mu_: torch.dtype, entropy penalty coefficient
        - lr: torch.dtype, learning rate
        - nb_iterations: integer, number of gradient steps
            
       Returns:
        - w: torch.tensor, regressor matrix 
    """
    
    w = torch.zeros(lon_size*lat_size,lon_size*lat_size, dtype=dtype)
    w_old = torch.zeros(lon_size*lat_size,lon_size*lat_size, dtype=dtype)
    
    # run a simple loop
    for it in range(nb_iterations):


        # accelerate gradient descent
        if it > 1:
            w_tmp = w + ((it-1)/(it+2)) * (w - w_old)
        else:
            w_tmp = w.detach()

        # save old parameter
        w_old = w.clone().detach()

        # compute gradient
        grad = compute_gradient(models,x,y,w_tmp,notnan_idx,lambda_,mu_,dtype=dtype)

        # update the variable w
        w = w_tmp - lr * grad

        # low-rank projection
        if rank is not None:
            w = low_rank_projection(w,rank=rank,dtype=dtype)


        if verbose==True:
            
            # compute loss functon to check convergence 
            res = torch.zeros(len(models))

            for idx_m, m in enumerate(models):

                # compute residuals
                res[idx_m] = torch.mean(torch.norm(y[m][:,:,notnan_idx] - x[m][:,:,notnan_idx] @ w[notnan_idx,:][:,notnan_idx], p='fro',dim=(1,2))**2)
        
            obj = mu_*torch.logsumexp((1/mu_)* res,0)
            obj += lambda_*torch.norm(w,p='fro')**2

            print("Iteration ", it,  ": Loss function : ", obj.item())
            
    return w


# function to compute weights
def compute_weights(models,w,x,y,notnan_idx,mu_=1.0,dtype=torch.float32):
    """Compute weights of models given regressor matrix W.
        
        Args:
            - models: list of strings
            - w: torch.tensor (grid_size, grid_size)
            - x,y: dictionaries of input-output pairs per model
            - notnan_idx: list of integers
            - lambda_: torch.dtype, ridge penalty coefficient
            - mu_: torch.dtype, entropy penalty coefficient
            

        Returns:
            - weights: dictionary of weights
    """
   
    M = len(list(models))
    alpha = torch.zeros(M,dtype=dtype)
    res = torch.zeros(M,dtype=dtype)
    
    for idx_m,m in enumerate(models):
        
        res[idx_m] = torch.mean(torch.norm(y[m][:,:,notnan_idx] - x[m][:,:,notnan_idx] @ w[notnan_idx,:][:,notnan_idx], p='fro',dim=(1,2))**2,dtype=dtype)
        alpha[idx_m] = (1/mu_)*res[idx_m]

    # softmax function to compute weights $\alpha$
    alpha = torch.nn.functional.softmax(alpha, dim=0, dtype=dtype)
    weights = {m: alpha[idx_m].item() for idx_m,m in enumerate(models)}

    return weights


############## Functions to optimize ridge penalty and trace norm penalty ##############
# 4 - $\min_{W} \Vert Y - X W \Vert_F^2 + \lambda \Vert W \Vert_F^2 + \nu \Vert W \Vert_{*}$
# optimal solution: $W = S_{nu/lambda}((X^T X + \lambda I)^{-1} X^T Y)$

def singular_value_thresholding(M, nu_):
    """Singular Value Thresholding (SVT) operator: M -> U * S_nu * V^T"""
    U, S, V = torch.svd(M)
    S_nu = torch.clamp(S - nu_, min=0)  # Soft-thresholding on singular values
    return U @ torch.diag(S_nu) @ V.t()

def train_trace_norm(X,Y,lambda_,nu_,dtype=torch.float32,verbose=False):
    """Compute the proximal operator of the elastic net penalty.
       argmin_(W) 1/2 ||Y - XW||_F^2 + lambda_ ||W||_F^2 + nu_ ||W||_* = 
       SingValue Soft-thresholding( (lambda I + X^T X )^{-1} X^T Y, nu_/lambda ))
    """
    W = ridge_regression(X,Y,lambda_,dtype=dtype,verbose=verbose)
    return singular_value_thresholding(W, nu_/lambda_)

######### Functions to optimize robust weight model with ridge penalty and trace norm penalty  #########
# 5 - $\min_{W} \max_{\alpha \in \Delta} \sum_{m} \alpha_m \Vert \Sigma^{-1/2}(Y_m - X_m W) \Vert_F^2 + \lambda \Vert W \Vert_F^2 + \nu \Vert W \Vert_{*}$ 
# we use proximal gradient descent to solve this problem

def frobenius_prox(x,lambda_):
    """Proximal operator for the Frobenius norm"""
    return x / (1 + lambda_)

def soft_thresholding(x,lambda_):
    """Soft-thresholding operator"""
    return torch.sign(x) * torch.max(torch.abs(x) - lambda_, torch.zeros_like(x))

def frobenius_and_trace_norm_prox(x,lambda_, nu_):
    """Proximal operator for the nuclear norm"""
    U, S, V = torch.svd(x)
    S = soft_thresholding(S,nu_)
    S = frobenius_prox(S,lambda_)
    return U @ torch.diag(S) @ V.t()

def compute_gradient_logsumexp(models,x,y,w,notnan_idx,mu_=1.0,dtype=torch.float32):
    """This function computes the gradient of ridge log-sum-exp loss with respect to W + ridge regularization + trace norm rgularizer.

    Args:
        - x, y: input-output pair
        - w: regressor matrix
        - B: positive definite matrix used in the variation 
        
    Returns:
        - Gradient matrix: torch.tensor d x d
    """
    res = torch.zeros(len(models), w.shape[0], w.shape[0]).to(dtype)
    res_sumexp = torch.zeros(len(models)).to(dtype)

    for idx_m, m in enumerate(models):

        # compute -2X_{m,r}^T (Y_{m,r}^T - X_{m,r}^T W)
        res[idx_m][np.ix_(notnan_idx,notnan_idx)] = - 2*torch.mean(torch.bmm(torch.transpose(x[m][:,:,notnan_idx], 1,2) , \
                                                        y[m][:,:,notnan_idx] - x[m][:,:,notnan_idx] @ w[np.ix_(notnan_idx,notnan_idx)]),dim=0)

        # compute the exponential term
        res_sumexp[idx_m] = (1/mu_)*torch.mean(torch.norm(y[m][:,:,notnan_idx] - x[m][:,:,notnan_idx] @ w[np.ix_(notnan_idx,notnan_idx)],p='fro',dim=(1,2))**2)
            
    softmax = torch.nn.Softmax(dim=0)
    res_sumexp = softmax(res_sumexp)

    # compute gradient as sum (res * softmax)
    grad = torch.sum(torch.unsqueeze(torch.unsqueeze(res_sumexp,-1),-1) * res, dim=0)
    
    return grad 

def train_robust_weights_trace_norm(models,x,y,lon_size,lat_size,notnan_idx,lambda_=1.0,mu_=1.0,nu_=1.0,lr=0.1,nb_iterations=10, dtype=torch.float32, verbose=False):
    """This function runs the (accelerated) proximal gradient algorithm to solve the optimization problem min f(W) + g(W) 
      where f(W) = mu * log (sum_m exp (1/mu * ||Y^m - X^m W||_F^2))  and g(W) = lambda * ||W||_F^2 + nu * ||W||_*

    Args:
        - models: list of models
        - x, y: input-output pair
        - notnan_idx: indices of the non missing values
        - lambda_: ridge penalty coefficient
        - nu_: trace norm penalty coefficient
        - mu_: temperature parameter
        - lr: learning rate
        - nb_prox_iterations: number of iterations

    Returns:
        - w: optimal regressor matrix
        - training_loss: training loss
    """
    w = torch.zeros(lon_size*lat_size,lon_size*lat_size).to(dtype)

    for it in range(nb_iterations):

        # compute gradient
        grad = compute_gradient_logsumexp(models,x,y,w,notnan_idx,mu_)

        # update the variable w
        w_tmp = w - lr * grad

        # compute proximal operator of trace norm and frobenius norm
        w_new = frobenius_and_trace_norm_prox(w_tmp, lr*lambda_, lr*nu_)

        # acceleration step
        w = w_new + ((it-1)/(it+2)) * (w_new - w)

        # compute training loss
        res = torch.zeros(len(models))
        for idx_m, m in enumerate(models):
            res[idx_m] = torch.mean(torch.norm(y[m][:,:,notnan_idx] -x[m][:,:,notnan_idx] @ w[notnan_idx,:][:,notnan_idx], p='fro',dim=(1,2))**2,dtype=dtype)

        if verbose==True:
            obj = mu_*torch.logsumexp((1/mu_)* res,0)
            obj += lambda_*torch.norm(w,p='fro')**2
            obj += nu_*torch.norm(w,p='nuc')

            print("Iteration ", it,  ": Loss function : ", obj.item())
            print("Rank of w: ", torch.linalg.matrix_rank(w))

    return w

############# prediction tools #########
def prediction(x, W, notnan_idx,nan_idx,dtype=torch.float32):
    """
    Compute target prediction given time series x and regressor W.

    Args:
        - x: torch.tensor (runs, time series length, grid size) 
        - W: torch.tensor (grid size non-nan idx, grid size non-nan idx)
        - notnan_idx, nan_idx: torch.tensor integers

    Returns:
        - y_pred: torch.tensor (runs, time series length, grid size)
    """    
    y_pred = torch.nan_to_num(x) @  W
    y_pred[:,:,nan_idx] = float('nan')
    
    return y_pred

