import xarray as xr
import pandas as pd
import numpy as np

# convert numpy arrays to xarray DataArrays
# convert numpy arrays to xarray DataArrays
def to_xarray(data, model_names, longitude, latitude):
    time = pd.date_range(start='1850-01-01', periods=data.shape[1], freq='ME')
    data_xr = xr.DataArray(
        data,
        dims=['model', 'time', 'latitude','longitude'],
        coords={
            'model': model_names,
            'time': time,
            'longitude': longitude,
            'latitude': latitude
        }
    )

    lon_xr = xr.DataArray(
        longitude,
        dims=['lon'],
        coords={'lon': longitude}
    )
    
    lat_xr = xr.DataArray(
        latitude,
        dims=['lat'],
        coords={'lat': latitude}
    )
    
    return data_xr, lon_xr, lat_xr

