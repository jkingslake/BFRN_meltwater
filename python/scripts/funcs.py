from matplotlib import pyplot as plt
from urllib.request import urlretrieve
import numpy as np
import xarray as xr
import os
import rasterio
import rioxarray
import panel.widgets as pnw
from typing import Optional, Tuple

## Define a function for running fill-spill-merge

def fsm(dem_filename: str, 
        prefix: str = "fsm_results/rema_tests/test-3", 
        uniform_melt: Optional[float] = None, 
        melt_filename: Optional[str] = None, 
        sea_level: float = 0.0,
        path_to_fsm: str = "../../../Barnes2020-FillSpillMerge/build/fsm.exe") -> xr.DataArray:
    """
    Runs the fill-spill-merge (FSM) algorithm on a digital elevation model (DEM) to calculate water depth.

    Parameters:
    - dem_filename (str): The filename of the DEM.
    - prefix (str): The prefix for the output files.
    - uniform_melt (float, optional): The uniform melt value (default: None).
    - melt_filename (str, optional): The filename of the melt input file (default: None).
    - sea_level (float): The sea level value (default: 0.0).
    - path_to_fsm (str): The path to the FSM executable (default: "../../../Barnes2020-FillSpillMerge/build/fsm.exe").

    Returns:
    - xr.DataArray: The water depth as a DataArray.

    Raises:
    - ValueError: If neither uniform_melt nor melt_filename is specified.
    
    Other:
    - This function is a wrapper for the fill-spill-merge algorithm.
    - The fill-spill-merge code also write the resulting water depth to a file ending with "-wtd.tif".
    """

    print("running fill-spill-merge...")
    

    if uniform_melt is not None:
        os.system(f"{path_to_fsm} {dem_filename} {prefix} {sea_level} --swl={uniform_melt}")

    if melt_filename is not None:
        os.system(f"{path_to_fsm} {dem_filename} {prefix} {sea_level} --swf={melt_filename}")

    if uniform_melt is None and melt_filename is None:
        raise ValueError("Must specify either uniform_melt or melt_filename") 


    print("loading results...")
    #surface_height = rioxarray.open_rasterio(prefix + "surface-height.tif").squeeze()
    water_depth = rioxarray.open_rasterio(prefix + "-wtd.tif").squeeze()
    #labels = rioxarray.open_rasterio(prefix + "label.tif").squeeze()

    return water_depth

## Define two functions for creating a melt map 
def rectangular_melt_region(dem: xr.DataArray, 
                            melt_magnitude: float, 
                            xmin: float = 815000, 
                            xmax: float = 820000, 
                            ymin: float = 1.93e6, 
                            ymax: float = 1.935e6,
                            melt_filename = "rema_subsets/water_input_file_3.tif")\
      -> Tuple[xr.DataArray, str, Tuple[float, float, float, float]]:
    
    # starting with the DEM, make a melt file with a rectangular region of non-zero melt
    melt = dem.copy().squeeze()
    melt[:, :] = 0
    melt.loc[ymax:ymin, xmin:xmax] = melt_magnitude  # ymin and ymax are flipped because the y-axis is flipped
    
    # Save the melt file
    melt.rio.to_raster(melt_filename)

    melt_bounds = (xmin, ymin, xmax, ymax)
    return melt, melt_filename, melt_bounds

# call rectangular_melt_region to make a square melt region
def square_melt_region(dem: xr.DataArray, 
                       melt_magnitude: float, 
                       x_center_of_melt: float = 817500, 
                       y_center_of_melt: float = 1.9325e6, 
                       width: float = 5000,
                       melt_filename = "rema_subsets/water_input_file_3.tif")\
      -> Tuple[xr.DataArray, str, Tuple[float, float, float, float]]:
    
    xmin = x_center_of_melt - width/2
    xmax = x_center_of_melt + width/2
    ymin = y_center_of_melt - width/2
    ymax = y_center_of_melt + width/2
    melt, melt_filename, melt_bounds = rectangular_melt_region(dem, melt_magnitude, xmin, xmax, ymin, ymax, melt_filename)
    
    return melt, melt_filename, melt_bounds


## Run FSM with different melt magnitudes
    
def loop_over_melt_magnitudes(dem_filename = "rema_subsets/dem_small_2.tif",
                            x_center_of_melt: float = 817500,
                            y_center_of_melt: float = 1.9325e6,
                            melt_width: float = 5000,                            
                            xmin: float = 815000, 
                            xmax: float = 820000, 
                            ymin: float = 1.93e6, 
                            ymax: float = 1.935e6,
                            iterations = 1,
                            start_melt_magnitude = 1,
                            end_melt_magnitude = 30,):
    # Load the DEM
    dem = rioxarray.open_rasterio(dem_filename, chunks={})
    dem = dem.squeeze()

    # Create a list of melt_magnitudes to loop over
    melt_magnitudes = np.linspace(start=start_melt_magnitude, stop=end_melt_magnitude, num=iterations)

    print(f"running fsm for melt_magnitudes: {melt_magnitudes} m")
    water_depths = []    # a list for holding multiple water_depth xr.DataArrays
    melts = []           # a list for holding multiple melt xr.DataArrays 
    bounds_list = []     # a list for holding the bounds of the rectangular melt regions 
    x_center_of_melts = []   # a list for holding the x center of the melt region (for when we define a square region)
    y_center_of_melts = []   # a list for holding the y center of the melt region (for when we define a square region)
    melt_widths = []      # a list for holding the width of the melt region (for when we define a square region)

    # loop over the melt_magnitudes 
    for melt_magnitude in melt_magnitudes:
        #melt, melt_filename, bounds = rectangular_melt_region(dem, melt_magnitude, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)   # write a melt tiff file to disk with a rectangular region of non-zero melt. Also return the melt xr.DataArray and the filename of the melt tiff file.
        melt, melt_filename, bounds = square_melt_region(dem, 
                                                         melt_magnitude, 
                                                         x_center_of_melt=x_center_of_melt, 
                                                         y_center_of_melt=y_center_of_melt, 
                                                         width=melt_width)  

        bounds_list.append(bounds)                                                   # append the bounds of the rectangular melt region to the list 'bounds_list'
        melts.append(melt)                                                           # append the melt xr.DataArray to the list 'melts'
        water_depths.append(fsm(dem_filename, melt_filename=melt_filename))          # append the water_depth xr.DataArray to the list 'water_depths'
        x_center_of_melts.append(x_center_of_melt)                                          # append the x center of the melt region to the list 'x_melt_center'
        y_center_of_melts.append(y_center_of_melt)                                          # append the y center of the melt region to the list 'y_melt_center'
        melt_widths.append(melt_width)                                                # append the width of the melt region to the list 'melt_width'

    # Create a new xarray from the array 'melt_magnitudes' so we can pass both the values of melt_magnitudes and the name 'melt_magnitudes' to xr.concat 
    melt_magnitudes_xr = xr.DataArray(melt_magnitudes, dims='melt_mag', name='melt_mag') 

    # concatenate
    water_depth = xr.concat(water_depths, dim=melt_magnitudes_xr)
    melt = xr.concat(melts, dim=melt_magnitudes_xr)

    # name the xr.DataArrays
    water_depth.name = 'water_depth'
    dem.name = 'dem'
    melt.name = 'melt'

    # merge the xr.DataArrays into a xr.Dataset
    results = xr.merge([water_depth, dem, melt])
    results = results.drop_vars('band')   # drop this unneeded variable

    # add information about the coordinates and variabels in attributes
    results.melt_mag.attrs = {'units': 'meters', 'long_name': 'melt magnitude', 'description': 'the magnitude of the melt in the rectangular region'}
    results.melt.attrs = {'units': 'meters', 'long_name': 'surface melt', 'description': 'the surface melt as a function of x and y'}

    # put bounds list into an xarray
    bounds_list = np.array(bounds_list)
    results['bounds'] = xr.DataArray(bounds_list, dims=['melt_mag', 'bounds_index'], name='bounds')
    results.bounds.attrs = {'long_name': 'bounds of the rectangular melt region', 'description': 'the bounds of the rectangular melt region: (xmin, ymin, xmax, ymax)'}

    # put x_melt_center list into an xarray
    x_center_of_melts = np.array(x_center_of_melts)
    results['x_center_of_melt'] = xr.DataArray(x_center_of_melts, dims='melt_mag', name='x_melt_center')
    results.x_center_of_melt.attrs = {'long_name': 'x coordinate of the center of the melt region', 'description': 'the x coordinate of the center of the melt region'}
    y_center_of_melts = np.array(y_center_of_melts) 
    results['y_center_of_melt'] = xr.DataArray(y_center_of_melts, dims='melt_mag', name='y_melt_center')
    results.y_center_of_melt.attrs = {'long_name': 'y coordinate of the center of the melt region', 'description': 'the y coordinate of the center of the melt region'}
    melt_widths = np.array(melt_widths)
    results['melt_width'] = xr.DataArray(melt_widths, dims='melt_mag', name='melt_width')
    results.melt_width.attrs = {'long_name': 'width of the melt region', 'description': 'the width of the melt region'}

    # add the center of mass of the water (see centroid_test.ipynb for notes on ths method)
    weights = results.water_depth.fillna(0)
    results['x_center_of_mass'] = results.x.weighted(weights).mean(dim = ['x', 'y'])
    results['y_center_of_mass'] = results.y.weighted(weights).mean(dim = ['x', 'y'])
    results.x_center_of_mass.attrs = {'long_name': 'x coordinate of the center of mass', 'description': 'the x coordinate of the center of mass of the water, i.e. the depth-weighted centroid'}
    results.y_center_of_mass.attrs = {'long_name': 'y coordinate of the center of mass', 'description': 'the y coordinate of the center of mass of the water, i.e. the depth-weighted centroid'}
    x_center_of_melt = (results['bounds'][:,2]+results['bounds'][:,0])/2
    y_center_of_melt = (results['bounds'][:,3]+results['bounds'][:,1])/2
    results['L'] = ((results['x_center_of_mass'] - results['x_center_of_melt'])**2 + (results['y_center_of_mass'] - results['y_center_of_melt'])**2)**(1/2)
    results.L.attrs = {'long_name': 'distance between the center of mass and the center of the melt region', 'description': 'the distance between the center of mass and the center of the melt region'}
    return results


def map_water_depth(results, coarsen_x=10, coarsen_y=10):
    coarse = results.coarsen(x=coarsen_x, y=coarsen_y, boundary='trim').mean()
    plot = coarse.water_depth.hvplot(x ='y', y = 'x', cmap='Blues', clim=(0,1), width = 1200, height = 300, aspect='equal')\
        * coarse.dem.hvplot.contour(x ='y', y = 'x',levels = 40, cmap='hot')\
        * coarse.melt.hvplot.contour(x ='y', y = 'x', levels=[0.0, 0.0])\
        * coarse.hvplot.scatter(x = 'y_center_of_mass', y = 'x_center_of_mass', color = 'green', size = 400, marker = '*')
    return plot