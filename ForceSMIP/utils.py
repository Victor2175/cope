import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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


def index_to_latlon(idx, longitude, latitude):
    lat_idx = idx // longitude.shape[0]
    lon_idx = idx %  longitude.shape[0]
    return lat_idx, lon_idx


EARTH_RADIUS_M = 6_371_000.0

def generate_lat_lon_grid(res_deg: float = 2.5,
                          lon_domain: str = "0-360"):
    """
    Generate regular lat/lon grid (cell centers + edges) at given resolution.

    Args:
        res_deg: Grid resolution in degrees (e.g. 2.5).
        lon_domain: "0-360" or "-180-180".

    Returns:
        lat_centers: (n_lat,) array from -90+res/2 to 90-res/2
        lon_centers: (n_lon,) array
        lat_edges: (n_lat+1,) array
        lon_edges: (n_lon+1,) array
    """
    # Latitude centers (exclude poles as centers; they are at ±90 edges)
    lat_centers = np.arange(-90 + res_deg/2, 90, res_deg)
    if lon_domain == "0-360":
        lon_centers = np.arange(0 + res_deg/2, 360, res_deg)
    else:
        lon_centers = np.arange(-180 + res_deg/2, 180, res_deg)

    # Edges
    lat_edges = np.concatenate(([-90.0], lat_centers[:-1] + res_deg/2, [90.0]))
    lon_edges = np.concatenate((
        [lon_centers[0] - res_deg/2],
        lon_centers[:-1] + res_deg/2,
        [lon_centers[-1] + res_deg/2]
    ))
    # Normalize lon edges
    if lon_domain == "0-360":
        lon_edges = np.mod(lon_edges, 360.0)
        # Ensure monotonic increasing (wrap final if needed)
        if not np.all(np.diff(np.sort(lon_edges)) > 0):
            lon_edges = np.linspace(0, 360, lon_centers.size + 1)
    else:
        # Force range -180..180
        lon_edges = ((lon_edges + 180) % 360) - 180
        if not np.all(np.diff(np.sort(lon_edges)) > 0):
            lon_edges = np.linspace(-180, 180, lon_centers.size + 1)

    return lat_centers, lon_centers, lat_edges, lon_edges

def compute_cell_areas(lat_edges: np.ndarray,
                       lon_edges: np.ndarray,
                       radius_m: float = EARTH_RADIUS_M) -> np.ndarray:
    """
    Compute spherical surface area (m^2) of each lat/lon cell.

    Args:
        lat_edges: (n_lat+1,) degrees
        lon_edges: (n_lon+1,) degrees
        radius_m: Sphere radius.

    Returns:
        areas: (n_lat, n_lon) array of cell areas.
    """
    # Convert to radians
    lat_rad = np.deg2rad(lat_edges)
    lon_rad = np.deg2rad(lon_edges)

    dlon = np.diff(lon_rad)[None, :]                     # (1, n_lon)
    # Spherical band area formula: R^2 * dlon * (sin φ_n - sin φ_s)
    sin_lat = np.sin(lat_rad)
    dsin = (sin_lat[1:] - sin_lat[:-1])[:, None]         # (n_lat, 1)

    areas = radius_m**2 * dlon * dsin                    # (n_lat, n_lon)
    return areas

def generate_grid_with_areas(res_deg: float = 2.5,
                             lon_domain: str = "0-360"):
    """
    Convenience wrapper returning centers, edges, areas.
    """
    lat_c, lon_c, lat_e, lon_e = generate_lat_lon_grid(res_deg, lon_domain)
    areas = compute_cell_areas(lat_e, lon_e)
    return lat_c, lon_c, lat_e, lon_e, areas  # shapes: (72),(144),(73),(145),(72,144)

def plot_empty_grid(res_deg: float = 2.5,
                    lon_domain: str = "0-360",
                    coastlines: bool = True,
                    figsize=(11,5),
                    central_longitude: float = 180):
    """
    Plot the 2.5° grid cell outlines.
    """
    lat_c, lon_c, lat_e, lon_e, _ = generate_grid_with_areas(res_deg, lon_domain)

    # Build mesh for pcolormesh (needs edge grids)
    Lon_e, Lat_e = np.meshgrid(lon_e, lat_e)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection=ccrs.Robinson(central_longitude=central_longitude))
    ax.set_global()
    if coastlines:
        ax.coastlines(linewidth=0.5)

    # Dummy zeros just to show grid
    ax.pcolormesh(Lon_e, Lat_e, np.zeros((lat_c.size, lon_c.size)),
                  transform=ccrs.PlateCarree(),
                  edgecolors='k', linewidth=0.15, shading='flat', cmap='Greys')

    ax.set_title(f"{res_deg}° x {res_deg} Global Grid ({lon_domain})", fontsize=12)
    return fig