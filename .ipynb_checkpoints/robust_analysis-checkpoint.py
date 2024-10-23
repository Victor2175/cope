import pickle
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as netcdf
import torch


def ridge_estimator(x,y,var,lambda_):
    """
    Compute the ridge estimator given gammas.
    """

    D = (1/var) * torch.eye(x.shape[0]).to(torch.float64)
    A = torch.matmul(torch.matmul(x.T, D),x) + lambda_ * torch.eye(x.shape[1])
    b = torch.matmul(torch.matmul(x.T,D),y)
    
    return torch.linalg.solve(A,b)


def train_ridge_regression(x,y,vars,lon_size,lat_size,models,lambda_,nbEpochs=100,verbose=True):
    """
    Given a model m, learn parameter β^m such that β^m = argmin_{β}(||y_m - X_m^T β||^2) ).

    Args:
        - x, y: training set and training target 
        - lon_size, lat_size: longitude and latitude grid size (Int)
        - lambda_: regularizer coefficient (float)
        - nbepochs: number of optimization steps (Int)
        - verbose: display logs (bool)
    """

    # define variable beta
    beta = torch.zeros(lon_size*lat_size).to(torch.float64)
    beta.requires_grad_(True)  
                          
    # define optimizer
    optimizer = torch.optim.Adam([beta],lr=1e-3)

    # stopping criterion
    criteria = torch.tensor(0.0)
    criteria_tmp = torch.tensor(1.0) 
    epoch = 0
            
    # --- optimization loop ---                
    while epoch < nbEpochs:

        # update criteria
        criteria_tmp = criteria.clone()
        epoch +=1
                      
        optimizer.zero_grad()
        ############### Define loss function ##############
                    
        # first term: ||Y - X - Rb ||
        res = torch.zeros(len(models),33)
       
        for idx_m, m in enumerate(models):
            for idx_i, i in enumerate(x[m].keys()):
                res[idx_m,:] += (y[m][i] - torch.matmul(x[m][i],beta))**2/vars[m]
            res[idx_m,:] = res[idx_m,:]/len(x[m].keys())

        obj = torch.mean(res)
        obj += lambda_*torch.norm(beta,p=2)**2
                    
        #define loss function
        loss = obj
                    
        # Use autograd to compute the backward pass. 
        loss.backward(retain_graph=True)               
        
        # take a step into optimal direction of parameters minimizing loss
        optimizer.step()       
        
        if(verbose==True):
            if(epoch % 10 == 0):
                print('Epoch ', epoch, 
                    ', loss=', loss.detach().item()
                    )

        criteria = loss

    # compute the alphas of the robust model
    M = len(x.keys())
    alpha = torch.zeros(M)
    res = torch.zeros(M,33)
    
    # compute the training loss for each model
    model_loss = {}
    
    for idx_m,m in enumerate(x.keys()):
        for idx_i, i in enumerate(x[m].keys()):
            res[idx_m,:] += (y[m][i] - torch.matmul(x[m][i],beta))**2/vars[m]
            
        res[idx_m,:] = res[idx_m,:]/len(x[m].keys())
        model_loss[m] = torch.mean(res[idx_m,:])
        
    return beta.detach().clone(), model_loss


def train_robust_model(x,y,vars,lon_size,lat_size,models,lambda_,mu_=1.0,nbEpochs=300,verbose=True):
    """
    Learn parameter β such that β = argmin( log Σ_m exp(||y_m - X_m^T β||^2) ).

    Args:
        - x,y : location, observation 
        - lon_size, lat_size: longitude and latitude grid size (Int)
        - models: (sub)list of models (list)
        - mu_: softmax coefficient (float)
        - nbepochs: number of optimization steps (Int)
        - verbose: display logs (bool)
    """

    # define variable beta
    beta = torch.zeros(lon_size*lat_size).to(torch.float64)
    beta.requires_grad_(True)  
                          
    # define optimizer
    optimizer = torch.optim.Adam([beta],lr=1e-4)

    # stopping criterion
    criteria = torch.tensor(0.0)
    criteria_tmp = torch.tensor(1.0) 
    epoch = 0
    training_loss = torch.zeros(nbEpochs)
            
    # --- optimization loop ---                
    while epoch < nbEpochs:

        # update criteria
        criteria_tmp = criteria.clone()
                      
        optimizer.zero_grad()
        
        ############### Define loss function ##############
                
        res = torch.zeros(len(models),33)

        for idx_m, m in enumerate(models):
            for idx_i, i in enumerate(x[m].keys()):
                res[idx_m,:] += (y[m][i] - torch.matmul(x[m][i],beta))**2/vars[m]
            res[idx_m,:] = res[idx_m,:]/len(x[m].keys())
            
        obj = mu_*torch.logsumexp((1/mu_)* torch.mean(res,axis=1),0)

        obj += lambda_*torch.norm(beta,p=2)**2
                    
        # define loss function
        loss = obj

        # set the training loss
        training_loss[epoch] = loss.detach().item()
                    
        # Use autograd to compute the backward pass. 
        loss.backward(retain_graph=True)               
        
        # take a step into optimal direction of parameters minimizing loss
        optimizer.step()       
        
        if(verbose==True):
            if(epoch % 10 == 0):
                print('Epoch ', epoch, 
                    ', loss=', training_loss[epoch].detach().item()
                    )
        criteria = loss
        epoch +=1

    # compute the alphas of the robust model
    M = len(x.keys())
    alpha = torch.zeros(M)
    res = torch.zeros(M,33)
    
    # compute the training loss for each model
    model_loss = {}
    
    for idx_m,m in enumerate(x.keys()):
        for idx_i, i in enumerate(x[m].keys()):
            res[idx_m,:] += (y[m][i] - torch.matmul(x[m][i],beta))**2/vars[m]
            
        res[idx_m,:] = res[idx_m,:]/len(x[m].keys())
        alpha[idx_m] = (1/mu_)*torch.mean(res[idx_m,:],axis=0)
        model_loss[m] = torch.mean(res[idx_m,:])
    
    alpha = torch.nn.functional.softmax(alpha)
    
    return beta.detach().clone(), alpha, model_loss


def compute_weights(x,y,vars,beta,lon_size,lat_size,models,mu_):
    """
    Plot and return the weights of the robust model.
    """
    M = len(list(x.keys()))
    gamma = torch.zeros(M)
    res = torch.zeros(M,33)
    
    for idx_m,m in enumerate(x.keys()):
        
        for idx_i, i in enumerate(x[m].keys()):
            res[idx_m,:] += (y[m][i] - torch.matmul(x[m][i],beta))**2/vars[m]
        res[idx_m,:] = res[idx_m,:]/len(x[m].keys())
        gamma[idx_m] = (1/mu_)*torch.mean(res[idx_m,:],axis=0)


    gamma = torch.nn.functional.softmax(gamma)

    # plot the model contributions
    weights = {m: gamma[idx_m].item() for idx_m,m in enumerate(models)}

    return weights

def leave_one_out(model_out,x,y,vars,lon_size,lat_size,lambda_,method='robust',mu_=1.0,nbEpochs=500,verbose=True):

    # Data preprocessing
    x_train = {}
    y_train = {}

    x_test = {}
    y_test = {}
    selected_models = []

    training_loss = np.zeros(len(x.keys()))

    for idx_m,m in enumerate(x.keys()):
        if m != model_out:

            x_train[m] = {}
            y_train[m] = {}
            
            # selected models 
            selected_models.append(m)
            
            for idx_i, i in enumerate(x[m].keys()):
                
                
                x_train[m][i] = torch.from_numpy(np.nan_to_num(x[m][i]).reshape(x[m][i].shape[0],lon_size*lat_size)).to(torch.float64)
                y_train[m][i] = torch.from_numpy(y[m][i]).to(torch.float64)
        
        else:
            for idx_i, i in enumerate(x[model_out].keys()):
                x_test[i] = np.nan_to_num(x[model_out][i]).reshape(x[model_out][i].shape[0],lon_size*lat_size)            
                y_test[i] = y[model_out][i]

    # if method = robust, then we train the robust
    if method == 'robust':
        beta, alpha, model_loss = train_robust_model(x_train,y_train,vars,\
                                                    lon_size,lat_size,\
                                                    selected_models,\
                                                    lambda_,mu_,nbEpochs,verbose)

        
        for idx_m,m in enumerate(x.keys()):
            if m != model_out:
                training_loss[idx_m] = model_loss[m]
    else:
        beta, model_loss = train_ridge_regression(x_train,y_train,vars,\
                                    lon_size,lat_size,\
                                    selected_models,\
                                    lambda_,nbEpochs,verbose)

        # set training loss
        for idx_m,m in enumerate(x.keys()):
            if m != model_out:
                training_loss[idx_m] = model_loss[m]

    y_pred={}
    for idx_i, i in enumerate(x[model_out].keys()):
        y_pred[i] = np.dot(x_test[i],beta)

    if method == 'robust':
        weights = compute_weights(x_train,y_train,vars,beta,lon_size,lat_size,selected_models,mu_)
    else:
        weights = {m: (1/len(x.keys())) for m in x.keys()}

    return beta, y_pred, y_test, weights, training_loss



def leave_one_out_procedure(x,y,vars,lon_size,lat_size,lambda_,method='robust',mu_=1.0,nbEpochs=500,verbose=True):

    beta = {}
    y_pred = {}
    y_test = {}
    rmse = {}
    weights = {m: 0.0 for idx_m, m in enumerate(x.keys())}
    training_loss = {m: np.zeros(len(x.keys())) for idx_m, m in enumerate(x.keys())}
    
    for idx_m, m in enumerate(x.keys()):
        
        beta[m], y_pred[m], y_test[m], weights_tmp, training_loss[m] = leave_one_out(m,x,y,vars,lon_size,lat_size,lambda_,method,mu_,nbEpochs,verbose)

        rmse[m] = 0
        for idx_i, i in enumerate(x[m].keys()):
            rmse[m] += np.mean(((y_test[m][i] - y_pred[m][i])**2/vars[m]).detach().numpy())
        rmse[m] = rmse[m]/len(x[m].keys())
            
        # compute the weight when a single model is out 
        if method == 'robust':    
            for m_tmp in list(x.keys()):
                if m_tmp != m:
                    weights[m_tmp] += (1/(len(x.keys())))* weights_tmp[m_tmp]

        # print the rmse
        print('RMSE on model ', m, ' : ', rmse[m])

    
    ################# plot the weights #################
    fig, ax = plt.subplots()
    models = list(x.keys()) 
    weights_plot = list(weights.values()) 
    ax.bar(models, weights_plot,label='Model weights')
    ax.set_ylabel(r'weights $\gamma$')
    ax.set_title('cmip6 models')
    ax.legend()
    ax.set_xticklabels(models, rotation=-90)
    plt.tight_layout()
    plt.savefig("results/weights_"+method+"_"+str(mu_)+"_"+str(lambda_)+".eps", dpi=150)
    plt.show()

    ################# plot the rmse #################
    fig, ax = plt.subplots()
    models = list(x.keys()) 
    rmse_plot = list(rmse.values()) 
    ax.bar(models, rmse_plot,label='rmse')
    ax.set_ylabel(r'LOO')
    ax.set_title('LOO rmse')
    ax.legend()
    ax.set_xticklabels(models, rotation=-90)
    plt.tight_layout()
    plt.savefig("results/rmse_"+method+"_"+str(mu_)+"_"+str(lambda_)+".eps", dpi=150)
    plt.show()
    

    return beta, rmse, weights, training_loss


def cross_validation_loo(x,y,vars,lon_size,lat_size,lambda_range,method='robust',mu_range=np.array([0.1,1.0,10.0]),nbEpochs=500,verbose=True):

    # create the pytorch tensor 
    beta = {}
    rmse = {}
    weights = {}
    y_pred = {}
    y_test = {}
    training_loss = {}

    if method != 'robust':
        mu_range_tmp = np.array([1.0])
    
    # for each lambda:
    for idx_lambda, lambda_ in enumerate(lambda_range):

        for idx_mu, mu_ in enumerate(mu_range):

            print("Cross validation: (" + str(mu_)+", "+ str(lambda_)+ ")")

            beta_tmp, rmse_tmp, weights_tmp, training_loss_tmp = leave_one_out_procedure(x,y,vars,\
                                                                      lon_size,lat_size,\
                                                                      lambda_,method,mu_,\
                                                                      nbEpochs=nbEpochs,verbose=False)

            beta[(mu_,lambda_)] = beta_tmp
            rmse[(mu_,lambda_)] = rmse_tmp
            weights[(mu_,lambda_)] = weights_tmp
            training_loss[(mu_,lambda_)] = training_loss_tmp

    return beta, rmse, weights, training_loss