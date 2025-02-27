import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from algorithms import prediction


def plot_gt_vs_pred(y_truth,x, w, notnan_idx, nan_idx,lon_grid, lat_grid, time_idx=10):
    """
    Plot groundtruth vs prediction for a given time index.

    Args:


    No returns.
    """

    # compute prediction given x and w.
    y_pred = prediction(x, w, notnan_idx, nan_idx)

    # capture grid size
    lat_size = lat_grid.shape[0]
    lon_size = lon_grid.shape[1]

    # define tensor to plot (for the first run)
    y_to_plot_target = y_truth[time_idx,:].detach().numpy().reshape(lat_size,lon_size)
    y_to_plot_pred = y_pred[time_idx,:].detach().numpy().reshape(lat_size,lon_size)

    # reset figures to plot
    plt.close('all')
    
    fig0 = plt.figure(figsize=(24,16))           

    ax0 = fig0.add_subplot(2, 2, 1)        
    ax0.set_title(r'Groundtruth', size=7,pad=3.0)
    im0 = ax0.pcolormesh(lon_grid,lat_grid,y_to_plot_target,vmin=-1.0,vmax=2.0)
    plt.colorbar(im0, ax=ax0, shrink=0.3)
    ax0.set_xlabel(r'x', size=7)
    ax0.set_ylabel(r'y', size=7)
    
    ax0 = fig0.add_subplot(2, 2, 2)        
    ax0.set_title(r'Predictions', size=7,pad=3.0)
    im0 = ax0.pcolormesh(lon_grid,lat_grid,y_to_plot_pred,vmin=-1.0,vmax=2.0)
    plt.colorbar(im0, ax=ax0, shrink=0.3)
    ax0.set_xlabel(r'x', size=7)
    ax0.set_ylabel(r'y', size=7)

    plt.show()


def plot_robust_weights(weights):
    """
    Function that displays climate model weights.

    Args:
        - weights: dictionary, scalar values

    Returns:
        None
    """

    plt.close('all')
    
    # plot the model contributions
    fig, ax = plt.subplots()
    models = list(weights.keys())
    weights_plot = list(weights.values())
    
    ax.bar(training_models, weights_plot,label='Model weights')
    ax.set_ylabel(r'weights $\alpha$')
    ax.set_ylim(0.0,1.0)
    ax.set_title('CMIP6 models')
    ax.legend()
    ax.set_xticklabels(models, rotation=-90)
    plt.tight_layout()

    
    plt.show()


########## plot animated gif of a groundtruth vs prediction ####################

def animation_gt_vs_pred(y_truth,x, w, notnan_idx, nan_idx,lon_grid, lat_grid, run_idx=0, time_period=34,savefile=False):
    """
    Plot groundtruth vs prediction for a given time index.

    Args:


    No returns.
    """
    
    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 150  
    plt.ioff()

    # compute prediction given x and w.
    y_pred = prediction(x, w, notnan_idx, nan_idx)

    # capture grid size
    lat_size = lat_grid.shape[0]
    lon_size = lon_grid.shape[1]
    
    plt.close('all')
    fig0 = plt.figure(figsize=(24,16))
    
    ax0 = fig0.add_subplot(1, 2, 1)        
    ax0.set_title(r'Groundtruth', size=7,pad=3.0)
    ax0.set_xlabel(r'x', size=7)
    ax0.set_ylabel(r'y', size=7)
    
    ax1 = fig0.add_subplot(1, 2, 2)        
    ax1.set_title(r'Prediction', size=7,pad=3.0)
    ax1.set_xlabel(r'x', size=7)
    ax1.set_ylabel(r'y', size=7)
    
    # get groundtruth and prediction to plot (first run)
    y_to_plot_target_tmp = y_truth[run_idx,:,:].detach().numpy().reshape(lat_size,lon_size)
    y_to_plot_pred_tmp = y_pred[run_idx,:,:].detach().numpy().reshape(lat_size,lon_size)
   
    im0 = ax0.pcolormesh(lon_grid,lat_grid,y_to_plot_target_tmp,vmin=-1.0,vmax=2.0)
    im1 = ax1.pcolormesh(lon_grid,lat_grid,y_to_plot_pred_tmp,vmin=-1.0,vmax=2.0)
   
    def animate_maps(i):
    
        y_to_plot_target = y_truth[run_idx,i,:].detach().numpy().reshape(lat_size,lon_size)
        y_to_plot_pred = y_pred[run_idx,i,:].detach().numpy().reshape(lat_size,lon_size)
    
        im0 = ax0.pcolormesh(lon_grid,lat_grid,y_to_plot_target,vmin=-1.0,vmax=2.0)
        im1 = ax1.pcolormesh(lon_grid,lat_grid,y_to_plot_pred,vmin=-1.0,vmax=2.0)
        
    plt.colorbar(im0, ax=ax0, shrink=0.3)
    plt.colorbar(im1, ax=ax1, shrink=0.3)
    anim = animation.FuncAnimation(fig0, animate_maps, frames=time_period)

    # save animation
    if savefile==True:
        anim.save(filename='results/gt_vs_prediction_run_'+str(run_idx)+'.mp4',writer=animation.FFMpegWriter(fps=5))