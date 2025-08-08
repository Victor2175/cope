import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from typing import Optional, List, Tuple
from sklearn.decomposition import PCA


class ForceSMIPVisualizer:
    """Class for visualizing ForceSMIP results and comparisons."""
    
    def __init__(self, longitude: np.ndarray, latitude: np.ndarray,
                 test_model_names: List[str]):
        self.longitude = longitude
        self.latitude = latitude
        self.test_model_names = test_model_names

    def plot_principal_component(self, w: np.ndarray, 
                                title: str = "Principal component of the weight matrix",
                                cmap: str = 'RdBu_r', 
                                vmin: Optional[float] = None, 
                                vmax: Optional[float] = None,
                                notnan_idx: Optional[List] = None,
                                nan_idx: Optional[List] = None,
                                n_components: int = 5, 
                                central_longitude: float = 180) -> plt.Figure:

        # Compute the SVD
        pca = PCA(n_components=n_components)
        pca.fit(w)

        # Get the first n_components principal components
        pcs_tmp = pca.components_
        pcs = np.zeros((pcs_tmp.shape[0], self.longitude.shape[0]*self.latitude.shape[0]), dtype=np.float32)
        pcs[:,notnan_idx] = pcs_tmp
        pcs[:,nan_idx] = np.nan  # Fill NaNs for visualization
        
        # Set color limits if not provided
        if vmin is None or vmax is None:
            vmax = np.nanmax(np.abs(pcs))
            vmin = -vmax


        for i in range(n_components):
            comp_ = pcs[i, :].reshape(self.latitude.shape[0], self.longitude.shape[0])

            # Create meshgrid for plotting
            lon_2d, lat_2d = np.meshgrid(self.longitude, self.latitude)

        
            # Create figure with Robinson projection
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_subplot(111, projection=ccrs.Robinson(central_longitude=central_longitude))
            
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
            
            # Plot the data
            levels = np.linspace(vmin, vmax, 20)
            im = ax.contourf(lon_2d, lat_2d, comp_,
                            levels, cmap=cmap,
                            transform=ccrs.PlateCarree(), extend='both')
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                            pad=0.1, shrink=0.8, aspect=30)
            cbar.set_label('Temperature Trend (°C/year)', fontsize=12)
            
            projection_type = "Pacific Centered" if central_longitude == 180 else "Atlantic Centered"
            plt.title(f'{title} - Principal component {i+1}\n(Robinson Projection - {projection_type}) \n Explained variance {pca.explained_variance_ratio_[i]:.2f}', 
                      fontsize=14, pad=20)
            plt.tight_layout()
        
        return fig


        return pcs

    def plot_robinson_projection(self, trend_data: np.ndarray, 
                                title: str = "Global Temperature Trend",
                                cmap: str = 'RdBu_r', 
                                vmin: Optional[float] = None, 
                                vmax: Optional[float] = None,
                                run_idx: int = 0, 
                                central_longitude: float = 180) -> plt.Figure:
        """
        Plot data using Robinson projection.
        
        Args:
            trend_data: Trend data array of shape (n_runs, n_grid_points)
            title: Plot title
            cmap: Colormap name
            vmin, vmax: Color scale limits
            run_idx: Which run to plot
            central_longitude: Longitude to center projection (180 for Pacific)
            
        Returns:
            Matplotlib figure
        """
        # Reshape trend data to 2D grid
        trend_2d = trend_data[run_idx, :].reshape(self.latitude.shape[0], self.longitude.shape[0])
        
        # Create meshgrid for plotting
        lon_2d, lat_2d = np.meshgrid(self.longitude, self.latitude)
        
        # Set color limits if not provided
        if vmin is None or vmax is None:
            vmax = np.nanmax(np.abs(trend_2d))
            vmin = -vmax

        print(f"Using vmin={vmin}, vmax={vmax} for colorbar")
        
        # Create figure with Robinson projection
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection=ccrs.Robinson(central_longitude=central_longitude))
        
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Plot the data
        levels = np.linspace(vmin, vmax, 20)
        im = ax.contourf(lon_2d, lat_2d, trend_2d,
                        levels, cmap=cmap,
                        transform=ccrs.PlateCarree(), extend='both')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                           pad=0.1, shrink=0.8, aspect=30)
        cbar.set_label('Temperature Trend (°C/year)', fontsize=12)
        
        projection_type = "Pacific Centered" if central_longitude == 180 else "Atlantic Centered"
        # plt.title(f'{title} - Model {self.test_model_names[run_idx]}\n(Robinson Projection - {projection_type})', 
        #           fontsize=14, pad=20)
        plt.tight_layout()
        
        return fig
    
    def plot_triple_comparison(self, trend_forced: np.ndarray, 
                              trend_prediction: np.ndarray,
                              trend_test_member: np.ndarray,
                              run_idx: int = 0,
                              central_longitude: float = 180,
                              cmap: str = 'RdBu_r',
                              vmin: float = -0.05,
                              vmax: float = 0.05) -> plt.Figure:
        """
        Plot three Robinson projections side by side for comparison.
        
        Args:
            trend_forced: Ground truth forced response trends
            trend_prediction: Model prediction trends
            trend_test_member: Raw test member trends
            run_idx: Which run to plot
            central_longitude: Longitude to center projection
            cmap: Colormap name
            vmin, vmax: Color scale limits
            
        Returns:
            Matplotlib figure
        """
        # Reshape all trend data to 2D grids
        trend_forced_2d = trend_forced[run_idx, :].reshape(self.latitude.shape[0], self.longitude.shape[0])
        trend_pred_2d = trend_prediction[run_idx, :].reshape(self.latitude.shape[0], self.longitude.shape[0])
        trend_test_2d = trend_test_member[run_idx, :].reshape(self.latitude.shape[0], self.longitude.shape[0])
        
        # Create meshgrid for plotting
        lon_2d, lat_2d = np.meshgrid(self.longitude, self.latitude)
        
        # Create figure with three subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Ground Truth (Forced Response)
        ax1 = fig.add_subplot(131, projection=ccrs.Robinson(central_longitude=central_longitude))
        ax1.set_global()
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax1.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax1.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        im1 = ax1.contourf(lon_2d, lat_2d, trend_forced_2d,
                           levels=20, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), extend='both')
        ax1.set_title(f'Ground Truth\n(Forced Response Trend)\nModel {self.test_model_names[run_idx]}', 
                      fontsize=14, pad=20)
        
        # 2. Model Prediction
        ax2 = fig.add_subplot(132, projection=ccrs.Robinson(central_longitude=central_longitude))
        ax2.set_global()
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax2.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax2.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)

        levels = np.linspace(vmin, vmax, 20)
        im2 = ax2.contourf(lon_2d, lat_2d, trend_pred_2d,
                           levels=levels, cmap=cmap,
                           transform=ccrs.PlateCarree(), extend='both')
        ax2.set_title(f'Model Prediction\n(Predicted Trend)\nModel {self.test_model_names[run_idx]}', 
                      fontsize=14, pad=20)
        
        # 3. Test Member (Raw Input)
        ax3 = fig.add_subplot(133, projection=ccrs.Robinson(central_longitude=central_longitude))
        ax3.set_global()
        ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax3.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax3.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax3.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        im3 = ax3.contourf(lon_2d, lat_2d, trend_test_2d,
                           levels=20, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), extend='both')
        ax3.set_title(f'Test Member\n(Raw Input Trend)\nModel {self.test_model_names[run_idx]}', 
                      fontsize=14, pad=20)
        
        # Add common colorbar
        cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], orientation='horizontal',
                           pad=0.1, shrink=0.8, aspect=40)
        cbar.set_label('Temperature Trend (°C/year)', fontsize=14)
        
        projection_type = "Pacific Centered" if central_longitude == 180 else "Atlantic Centered"
        # plt.suptitle(f'Temperature Trend Comparison - {self.test_model_names[run_idx]}\n(Robinson Projection - {projection_type})', 
                    #  fontsize=16, y=0.95)
        # plt.tight_layout()
        
        return fig
    
    def plot_time_series(self, data_dict: dict, lat_idx: int, lon_idx: int,
                        run_idx: int = 0, title: Optional[str] = None) -> plt.Figure:
        """
        Plot time series at a specific grid point.
        
        Args:
            data_dict: Dictionary with different data series
            lat_idx: Latitude index
            lon_idx: Longitude index 
            run_idx: Run index
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for label, data in data_dict.items():
            if len(data.shape) == 4:  # (runs, time, lat, lon)
                series = data[run_idx, :, lat_idx, lon_idx]
            elif len(data.shape) == 3:  # (runs, time, features) - flattened
                grid_idx = lat_idx * self.longitude.shape[0] + lon_idx
                series = data[run_idx, :, grid_idx]
            else:
                continue
                
            ax.plot(series, label=label, linewidth=2)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Temperature', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if title is None:
            title = f'Time Series at lat={self.latitude[lat_idx]:.1f}°, lon={self.longitude[lon_idx]:.1f}°'
        ax.set_title(title, fontsize=14)
        
        # plt.tight_layout()
        return fig