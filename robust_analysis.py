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

def train_ridge_regression(x,y,var,lon_size,lat_size,lambda_,nbEpochs=100,verbose=True):
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
    beta = torch.zeros(grid_lat_size*grid_lon_size).to(torch.float64)
    beta.requires_grad_(True)  
                          
    # define optimizer
    optimizer = torch.optim.Adam([beta],lr=1e-3)
            
    # --- optimization loop ---                
    for epoch in torch.arange(nbEpochs):      
                      
        optimizer.zero_grad()
        ############### Define loss function ##############
                    
        # first term: ||Y - X - Rb ||
        obj = 0.5*torch.mean((y - torch.matmul(x,beta))**2/var)
        obj += 0.5*lambda_*torch.norm(beta,p=2)**2
                    
        #define loss function
        loss = obj
                    
        # Use autograd to compute the backward pass. 
        loss.backward(retain_graph=True)               
        
        # take a step into optimal direction of parameters minimizing loss
        optimizer.step()       
        
        if(verbose==True):
            if(epoch % 10 == 0):
                print('Epoch ', epoch.detach().item(), 
                    ', loss=', loss.detach().item()
                    )
    return beta.detach().clone()


def train_robust_model(x,y,vars,lon_size,lat_size,models,alpha_,lambda_,nbEpochs,verbose=True):
    """
    Learn parameter β such that β = argmin( log Σ_m exp(||y_m - X_m^T β||^2) ).

    Args:
        - x: 
        - lon_size, lat_size: longitude and latitude grid size (Int)
        - models: (sub)list of models (list)
        - alpha_: softmax coefficient (float)
        - nbepochs: number of optimization steps (Int)
        - verbose: display logs (bool)
    """

    # define variable beta
    beta = torch.zeros(lon_size*lat_size).to(torch.float64)
    beta.requires_grad_(True)  
                          
    # define optimizer
    optimizer = torch.optim.Adam([beta],lr=1e-3)
            
    # --- optimization loop ---                
    for epoch in torch.arange(nbEpochs):      
                      
        optimizer.zero_grad()
        ############### Define loss function ##############
                    
        # first term: ||Y - X - Rb ||
        obj = torch.tensor(0.0)
        for idx_m,m in enumerate(models):
            obj += torch.exp((1/alpha_)*torch.mean((y[m] - torch.matmul(x[m],beta))**2)/vars[m])
    
        obj = alpha_*torch.log(obj)

        obj += lambda_*torch.norm(beta,p=2)**2
                    
        #define loss function
        loss = obj
                    
        # Use autograd to compute the backward pass. 
        loss.backward(retain_graph=True)               
        
        # take a step into optimal direction of parameters minimizing loss
        optimizer.step()       
        
        if(verbose==True):
            if(epoch % 10 == 0):
                print('Epoch ', epoch.detach().item(), 
                    ', loss=', loss.detach().item()
                    )
    return beta.detach().clone()


def compute_weights(m,x,y,vars,beta,lon_size,lat_size,alpha_,lambda_):
    """
    Plot and return the weights of the robust model.
    """
    
    # compute the coefficient using soft max
    M = len(list(x.keys()))
    gamma = np.zeros(M)
    beta_tmp = beta.detach().numpy()
    
    for idx_m,m_it in enumerate(x.keys()):
        if m_it != m:
            gamma[idx_m] = np.exp((1/alpha_)*np.mean((y[m_it].numpy() - np.dot(x[m_it].numpy(),beta_tmp))**2/vars[m_it]))
    
    gamma = gamma/np.sum(gamma)
    
    # plot the model contributions
    models = list(vars.keys())
    weights = {m: gamma[idx_m].item() for idx_m,m in enumerate(x.keys())}

    return weights


def leave_one_out(model_out,x,y,vars,lon_size,lat_size,alpha_,lambda_,nbEpochs=500,verbose=True):

    # Data preprocessing
    x_train = {}
    y_train = {}
    selected_models = []

    for idx_m,m in enumerate(x.keys()):
        if m != model_out:

            selected_models.append(m)
            
            x_train[m] = torch.from_numpy(np.nan_to_num(x[m]).reshape(x[m].shape[0],x[m].shape[1]*x[m].shape[2])).to(torch.float64)
            y_train[m] = torch.from_numpy(np.nan_to_num(y[m])).to(torch.float64)
        
            nans_idx = np.where(np.isnan(x[m][0,:,:].ravel()))[0]

        else:
            x_test = np.nan_to_num(x[m]).reshape(x[m].shape[0],x[m].shape[1]*x[m].shape[2])            
            y_test = np.nan_to_num(y[m])


    beta_robust = train_robust_model(x_train,y_train,vars,\
                                      lon_size,lat_size,\
                                      selected_models,alpha_,lambda_,nbEpochs,verbose)

    
    y_pred = np.dot(x_test,beta_robust)

    weights = compute_weights(model_out,x_train,y_train,vars,beta_robust,lon_size,lat_size,alpha_,lambda_)


    return beta_robust, y_pred, y_test, weights

def leave_one_out_procedure(x,y,vars,lon_size,lat_size,alpha_,lambda_,nbEpochs=500,verbose=True):

    beta_robust = {}
    y_pred = {}
    y_test = {}
    rmse = {}
    weights = {m: 0.0 for idx_m, m in enumerate(x.keys())}

    # define the file
    file = '/net/h2o/climphys3/simondi/cope-analysis/data/erss/sst_annual_g050_mean_19812014_centered.nc'
    
    # read the dataset
    file2read = netcdf.Dataset(file,'r')
    
    # load longitude, latitude and sst monthly means
    lon = np.array(file2read.variables['lon'][:])
    lat = np.array(file2read.variables['lat'][:])
    sst = np.array(file2read.variables['sst'])
    
    # define grid
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    for idx_m, m in enumerate(x.keys()):
        beta_robust[m], y_pred[m], y_test[m], weights_tmp = leave_one_out(m,x,y,vars,lon_size,lat_size,alpha_,lambda_,nbEpochs,verbose)
        rmse[m] =  np.mean((y_test[m] - y_pred[m])**2)

        nans_idx = np.where(np.isnan(x[m][0,:,:].ravel()))[0]

        # compute the weight when a single model is out 
        for m_tmp in list(x.keys()):
            if m_tmp != m:
                weights[m_tmp] += (1/(len(x.keys())-1))* weights_tmp[m_tmp]

        # print the rmse
        print('RMSE on model ', m, ' : ', np.mean((y_pred[m] - y_test[m])**2))

    # create the function y=x
    minx = np.min(y_test[m])
    maxx = np.max(y_test[m])
    x_tmp = np.linspace(minx,maxx,100)
    y_tmp = x_tmp

    ################# plot the observation vs prediction accuracy #####################################
    fig, axs = plt.subplots(6,5, figsize=(15,10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 2.0, wspace=1.0)

    axs = axs.ravel()
    
    for idx_m, m in enumerate(x.keys()):

        axs[idx_m].scatter(y_test[m],y_pred[m],label=m,s=0.1)
        axs[idx_m].plot(x_tmp,y_tmp,color='r',linewidth=0.5)
        axs[idx_m].set_title(m)

    for i in range(len(x.keys()),30):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.savefig("results/pred_vs_real_"+str(alpha_)+"_"+str(lambda_)+".eps", dpi=150)
    plt.show()

    ############################### plot the residuals #####################################
    fig, axs = plt.subplots(6,5, figsize=(15,10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 2.0, wspace=1.0)

    axs = axs.ravel()
    
    for idx_m, m in enumerate(x.keys()):

        axs[idx_m].scatter(y_test[m],y_test[m] - y_pred[m],label=m,s=0.1)
        axs[idx_m].plot(x_tmp,np.zeros_like(x_tmp),color='r',linewidth=0.5)
        axs[idx_m].set_title(m)

    for i in range(len(x.keys()),30):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.savefig("results/residuals_"+str(alpha_)+"_"+str(lambda_)+".eps", dpi=150)
    plt.show()
    
    ############## plot the beta map for each leave-one-out run #####################################
    fig, axs = plt.subplots(6,5, figsize=(15,10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 2.0, wspace=1.0)

    axs = axs.ravel()
    
    for idx_m, m in enumerate(x.keys()):
        
        beta_robust_tmp = beta_robust[m].detach().clone()
        beta_robust_tmp[nans_idx] = 1e5
        beta_robust_tmp = beta_robust_tmp.detach().numpy().reshape(lat_size,lon_size)

        axs[idx_m].set_title(m)
        im0 = axs[idx_m].pcolormesh(lon_grid,lat_grid,beta_robust_tmp,vmin=-0.00,vmax = 0.005)

    plt.colorbar(im0, ax=axs[idx_m], shrink=0.5)

    for i in range(len(x.keys()),30):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.savefig("results/beta_map_"+str(alpha_)+"_"+str(lambda_)+".eps", dpi=150)
    plt.show()

    
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
    plt.savefig("results/weights_"+str(alpha_)+"_"+str(lambda_)+".eps", dpi=150)
    plt.show()
    
    

    return beta_robust, rmse, weights

def cross_validation_loo(x,y,vars,lon_size,lat_size,\
                         alpha_range,lambda_range,\
                         nbEpochs=500,verbose=True):

    # create the pytorch tensor 
    beta_robust = {}
    rmse = {}
    weights = {}
    y_pred = {}
    y_test = {}
    
   
    # for each pair (alpha, lambda)
    for idx_alpha, alpha_ in enumerate(alpha_range):

        # for each lambda:
        for idx_lambda, lambda_ in enumerate(lambda_range):

            print("Cross validation: (" + str(alpha_)+", "+ str(lambda_)+ ")")

            beta_robust_tmp, rmse_tmp, weights_tmp = leave_one_out_procedure(x,y,vars,lon_size,lat_size,alpha_,lambda_,nbEpochs=nbEpochs,verbose=verbose)

            beta_robust[(alpha_,lambda_)] = beta_robust_tmp
            rmse[(alpha_,lambda_)] = rmse_tmp
            weights[(alpha_,lambda_)] = weights_tmp

    return beta_robust, rmse, weights

def ridge_estimator(model_out,x,y,vars,lambda_):
    """
    Compute the ridge estimator given gammas.
    """
    idx_start = 0
    for idx_m,m in enumerate(list(x.keys())):
        if m!= model_out:
            if idx_start==0:
                X_tmp = x[m]
                y_tmp = y[m]
                D = vars[m]*torch.eye(x[m].shape[0])
                idx_start +=1
            else:
                X_tmp = torch.cat((X_tmp,x[m]),0)
                y_tmp = torch.cat((y_tmp,y[m]),0)
                D_tmp = (vars[m] * torch.eye(x[m].shape[0])).to(torch.float64)
                D = torch.block_diag(D, D_tmp).to(torch.float64)

    A = torch.matmul(torch.matmul(X_tmp.T, D),X_tmp) + lambda_ * torch.eye(X_tmp.shape[1])
    b = torch.matmul(torch.matmul(X_tmp.T,D),y_tmp)
    
    return torch.linalg.solve(A,b)

def leave_one_out_ridge(model_out,x,y,vars,lambda_):

    # Data preprocessing
    x_train = {}
    y_train = {}
    selected_models = []

    for idx_m,m in enumerate(x.keys()):
        if m != model_out:

            selected_models.append(m)
            
            x_train[m] = torch.from_numpy(np.nan_to_num(x[m]).reshape(x[m].shape[0],x[m].shape[1]*x[m].shape[2])).to(torch.float64)
            y_train[m] = torch.from_numpy(np.nan_to_num(y[m])).to(torch.float64)
        
            nans_idx = np.where(np.isnan(x[m][0,:,:].ravel()))[0]

        else:
            x_test = np.nan_to_num(x[m]).reshape(x[m].shape[0],x[m].shape[1]*x[m].shape[2])            
            y_test = np.nan_to_num(y[m])


    beta_ridge = ridge_estimator(model_out,x_train,y_train,vars,lambda_)
    
    y_pred = np.dot(x_test,beta_ridge)

    return beta_ridge, y_pred, y_test

def leave_one_out_procedure_ridge(x,y,vars,lon_size,lat_size,lambda_):

    beta_ridge = {}
    y_pred = {}
    y_test = {}
    rmse = {}
    weights = {m: 0.0 for idx_m, m in enumerate(x.keys())}

    # define the file
    file = '/net/h2o/climphys3/simondi/cope-analysis/data/erss/sst_annual_g050_mean_19812014_centered.nc'
    
    # read the dataset
    file2read = netcdf.Dataset(file,'r')
    
    # load longitude, latitude and sst monthly means
    lon = np.array(file2read.variables['lon'][:])
    lat = np.array(file2read.variables['lat'][:])
    sst = np.array(file2read.variables['sst'])

    # define grid
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    for idx_m, m in enumerate(x.keys()):
        
        beta_ridge[m], y_pred[m], y_test[m] = leave_one_out_ridge(m,x,y,vars,lambda_)
        rmse[m] =  np.mean((y_test[m] - y_pred[m])**2)

        nans_idx = np.where(np.isnan(x[m][0,:,:].ravel()))[0]

        # print the rmse
        print('RMSE on model ', m, ' : ', np.mean((y_pred[m] - y_test[m])**2))

    # create the function y=x
    minx = np.min(y_test[m])
    maxx = np.max(y_test[m])
    x_tmp = np.linspace(minx,maxx,100)
    y_tmp = x_tmp

    ################# plot the observation vs prediction accuracy #####################################
    fig, axs = plt.subplots(6,5, figsize=(15,10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 2.0, wspace=1.0)

    axs = axs.ravel()
    
    for idx_m, m in enumerate(x.keys()):

        axs[idx_m].scatter(y_test[m],y_pred[m],label=m,s=0.1)
        axs[idx_m].plot(x_tmp,y_tmp,color='r',linewidth=0.5)
        axs[idx_m].set_title(m)

    for i in range(len(x.keys()),30):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.savefig("results/pred_vs_real_ridge_"+str(lambda_)+".eps", dpi=150)
    plt.show()

    ############################### plot the residuals #####################################
    fig, axs = plt.subplots(6,5, figsize=(15,10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 2.0, wspace=1.0)

    axs = axs.ravel()
    
    for idx_m, m in enumerate(x.keys()):

        axs[idx_m].scatter(y_test[m],y_test[m] - y_pred[m],label=m,s=0.1)
        axs[idx_m].plot(x_tmp,np.zeros_like(x_tmp),color='r',linewidth=0.5)
        axs[idx_m].set_title(m)

    for i in range(len(x.keys()),30):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.savefig("results/residuals_ridge_"+str(lambda_)+".eps", dpi=150)
    plt.show()
    
    ############## plot the beta map for each leave-one-out run #####################################
    fig, axs = plt.subplots(6,5, figsize=(15,10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 2.0, wspace=1.0)

    axs = axs.ravel()
    
    for idx_m, m in enumerate(x.keys()):
        
        beta_ridge_tmp = beta_ridge[m].detach().clone()
        beta_ridge_tmp[nans_idx] = 1e5
        beta_ridge_tmp = beta_ridge_tmp.detach().numpy().reshape(lat_size,lon_size)

        axs[idx_m].set_title(m)
        im0 = axs[idx_m].pcolormesh(lon_grid,lat_grid,beta_ridge_tmp,vmin=-0.00,vmax = 0.005)

    plt.colorbar(im0, ax=axs[idx_m], shrink=0.5)

    for i in range(len(x.keys()),30):
        fig.delaxes(axs[i])

    fig.tight_layout()
    plt.savefig("results/beta_ridge_map_"+str(lambda_)+".eps", dpi=150)
    plt.show()
    

    return beta_ridge, rmse

def cross_validation_loo_ridge(x,y,vars,lon_size,lat_size,lambda_range):

    # create the pytorch tensor 
    beta = {}
    rmse = {}
    weights = {}
    y_pred = {}
    y_test = {}
    
   
    for idx_lambda, lambda_ in enumerate(lambda_range):

        print("Cross validation: ("+ str(lambda_)+ ")")

        beta_ridge_tmp, rmse_tmp = leave_one_out_procedure_ridge(x,y,vars,lon_size,lat_size,lambda_)

        beta[lambda_] = beta_ridge_tmp
        rmse[lambda_] = rmse_tmp

    return beta, rmse