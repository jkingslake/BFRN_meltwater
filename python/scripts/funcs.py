from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import os
import rioxarray
from typing import Optional, Tuple, Callable
import subprocess
import hvplot.xarray # noqa 
from tqdm.autonotebook import tqdm
import pandas as pd
hvplot.extension('bokeh')
import sparse


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

    if uniform_melt is not None:
        subprocess.run([path_to_fsm, dem_filename, prefix, str(sea_level), "--swl=" + str(uniform_melt)], stdout=subprocess.DEVNULL)
        #os.system(f"{path_to_fsm} {dem_filename} {prefix} {sea_level} --swl={uniform_melt}")

    if melt_filename is not None:
        subprocess.run([path_to_fsm, dem_filename, prefix, str(sea_level), "--swf=" + melt_filename], stdout=subprocess.DEVNULL)
        #os.system(f"{path_to_fsm} {dem_filename} {prefix} {sea_level} --swf={melt_filename}")

    if uniform_melt is None and melt_filename is None:
        raise ValueError("Must specify either uniform_melt or melt_filename") 

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
                            end_melt_magnitude = 30,
                            plot_with_hvplot=False):
    """This function is an old approach to looping. Use gridSearch and fsm_xarray instead."""
    
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

    # add information about the coordinates and variables in attributes
    results.melt_mag.attrs = {'units': 'meters', 'long_name': 'melt magnitude', 'description': 'the magnitude of the melt in the rectangular region'}
    results.melt.attrs = {'units': 'meters', 'long_name': 'surface melt', 'description': 'the surface melt as a function of x and y'}

    # put bounds list into the results xr.Dataset
    bounds_list = np.array(bounds_list)
    results['bounds'] = xr.DataArray(bounds_list, dims=['melt_mag', 'bounds_index'], name='bounds')
    results.bounds.attrs = {'long_name': 'bounds of the rectangular melt region', 'description': 'the bounds of the rectangular melt region: (xmin, ymin, xmax, ymax)'}

    # put x_melt_center, y_melt_center, and melt_width into the results xr.Dataset
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
    #x_center_of_melt = (results['bounds'][:,2]+results['bounds'][:,0])/2
    #y_center_of_melt = (results['bounds'][:,3]+results['bounds'][:,1])/2
    results['L'] = ((results['x_center_of_mass'] - results['x_center_of_melt'])**2 + (results['y_center_of_mass'] - results['y_center_of_melt'])**2)**(1/2)
    results.L.attrs = {'long_name': 'distance between the center of mass and the center of the melt region', 'description': 'the distance between the center of mass and the center of the melt region'}
    
    if plot_with_hvplot:
        results.attrs['hvplot_function'] = map_water_depth(results)
    
    return results

def map_water_depth(results, coarsen_x=10, coarsen_y=10, width=500, height=500):
    coarse = results.coarsen(x=coarsen_x, y=coarsen_y, boundary='trim').mean()
    plot = coarse.water_depth.hvplot(x ='y', y = 'x', cmap='Blues', clim=(0,1), width = width, height = height, aspect='equal')\
        * coarse.dem.hvplot.contour(x ='y', y = 'x',levels = 40, cmap='hot')\
        * coarse.melt.hvplot.contour(x ='y', y = 'x', levels=[0.0, 0.0])\
        * coarse.hvplot.scatter(x = 'y_center_of_mass', y = 'x_center_of_mass', color = 'green', size = 400, marker = '*')
    return plot

def gridSearch(function: Callable, **kwargs) -> xr.core.dataset.Dataset:
    """
    Perform a grid search by iterating over all combinations of input parameters and running a given function.

    Parameters:
    function (callable): The function to be executed for each combination of input parameters. This function should return either an xarray dataset, an xarray datarray, or a numpy array.
    **kwargs: Keyword arguments representing the input parameters and their corresponding values.

    Returns:
    xr_unstacked (xarray.core.dataset.Dataset): The concatenated and unstacked xarray dataset containing the results of the grid search.

    Example:
    #### Define a function to be executed for each combination of input parameters
    def my_function(param1, param2):
        ##### Perform some computation using the input parameters
        result = param1 + param2
        return result

    #### Perform a grid search by iterating over all combinations of input parameters
    results = gridSearch(my_function, param1=[1, 2, 3], param2=[4, 5])
    
    """

    # extract the names of the parameters
    p_names = [x for x in kwargs] 

    # extract the values of the parameters
    p_values_list = [x for x in kwargs.values()]
    
    # p_values_list should be a list of lists, if any items are not a list, then this 
    # is because the user has only specified a single value for that parameter. 
    # Loop through the list and convert any items that are not a list into a list containing the one item.
    p_values_list = [[x] if not isinstance(x,list) else x for x in p_values_list]

    # create a multiIndex from the parameter names and values
    multiIndex = pd.MultiIndex.from_product(p_values_list, names=p_names)


    #loop over every conbimation of parameters stored in multiIndex
    xr_out_list = []
    for mi in tqdm(multiIndex):

        # create a dictionary of inputs for the function from the values stored in multiIndex
        inputs = {p_names[x]: mi[x] for x in range(len(p_names))}

        # run the function with this combination of inputs
        single_iteration_result = function(**inputs)

        # add coordinates to the result and store as as either a DataSet or a dataArray
        if isinstance(single_iteration_result, xr.core.dataset.Dataset):
            xr_out_new = single_iteration_result.assign_coords(inputs)
        else:
            xr_out_new = xr.DataArray(single_iteration_result, coords=inputs)    # use this line if the function returns a data array, or a numpy array
        
        # append the result to a list
        xr_out_list.append(xr_out_new)

    # concatenate the list of results into a single xarray
    xr_stacked = xr.concat(xr_out_list, dim='stacked_dim')

    # add the multiIndex to the xarray
    mindex_coords = xr.Coordinates.from_pandas_multiindex(multiIndex, 'stacked_dim')
    xr_stacked = xr_stacked.assign_coords(mindex_coords)

    # unstack the xarray - i.e. separate the multiIndex into separate dimensions
    xr_unstacked = xr_stacked.unstack()

    # convert to a dataset if the result is a data array
    if isinstance(xr_unstacked, xr.DataArray):
        xr_unstacked.name = 'result'
        xr_unstacked = xr_unstacked.to_dataset()


    return xr_unstacked


def add_center_of_mass(results, 
                       x_center_of_melt, 
                       y_center_of_melt):
    # add the center of mass of the water (see centroid_test.ipynb for notes on this method)
    weights = results.water_depth.fillna(0)
    results['x_center_of_mass'] = results.x.weighted(weights).mean(dim = ['x', 'y'])
    results['y_center_of_mass'] = results.y.weighted(weights).mean(dim = ['x', 'y'])
    results.x_center_of_mass.attrs = {'long_name': 'x coordinate of the center of mass', 'description': 'the x coordinate of the center of mass of the water, i.e. the depth-weighted centroid'}
    results.y_center_of_mass.attrs = {'long_name': 'y coordinate of the center of mass', 'description': 'the y coordinate of the center of mass of the water, i.e. the depth-weighted centroid'}
    results['L'] = ((results['x_center_of_mass'] - x_center_of_melt)**2 + (results['y_center_of_mass'] - y_center_of_melt)**2)**(1/2)
    results.L.attrs = {'long_name': 'distance between the center of mass and the center of the melt region', 'description': 'the distance between the center of mass and the center of the melt region'}
    return results

def GL_flux(results):

    bfrn = xr.open_dataset('/Users/jkingslake/Documents/science/meltwater_routing/BFRN_meltwater/data/BFRN/bfrn.nc')   #  BFRN_meltwater/data/BFRN/load_BFRN.ipynb generates this file
    bfrn = bfrn['__xarray_dataarray_variable__']
    bfrn.name = 'bfrn'
    
    bfrn_high_res = bfrn.interp_like(results, method='nearest')

    redistribution_depth = -results['melt'] + results['water_depth']
    redistribution_depth.attrs['units'] = 'm'
    redistribution_depth.attrs['long_name'] = 'Redistribution depth'

    x_resolution = results.x[1].values - results.x[0].values
    assert x_resolution == np.abs(results.y[1].values - results.y[0].values)
    resolution = x_resolution
    cell_area = resolution**2

    redistribution_volume_per_cell = redistribution_depth * cell_area
    redistribution_volume_per_cell.name = 'redistribution_volume_per_cell'
    redistribution_volume_per_cell.attrs['units'] = 'm3'
    redistribution_mass_per_cell = redistribution_volume_per_cell * 1000
    redistribution_mass_per_cell.name = 'redistribution_mass_per_cell'
    redistribution_mass_per_cell.attrs['units'] = 'kg'

    GL_flux_response_per_cell = -redistribution_mass_per_cell * bfrn_high_res
    GL_flux_response_per_cell.name = 'GL_flux_response_per_cell'
    GL_flux_response_per_cell.attrs['description'] = 'The change in grounding line flux due to the change in ice shelf thickness in each cell fue to water redistribution'
    GL_flux_response_per_cell.attrs['units'] = 'kg/yr'


    total_GL_flux_response = GL_flux_response_per_cell.sum(dim=['x','y'])
    total_GL_flux_response.name = 'total_GL_flux_response'
    total_GL_flux_response.attrs['units'] = 'kg/yr'
    total_GL_flux_response.attrs['description'] = 'The total instantaneous change in grounding line flux due to the change in ice shelf thickness in each cell due to water redistribution'

    return xr.merge([results, total_GL_flux_response])

def fsm_xarray(dem_filename="rema_subsets/dem_small_2.tif",
               melt_magnitude=0.1,
               x_center_of_melt: float = 817500.0,
               y_center_of_melt: float = 1.9325e6,
               melt_width: float = 5000,
               sparse: bool = True) -> xr.Dataset:
    """
    Perform meltwater routing using fill-spill-merge and output an xarray dataset.

    Parameters:
    - dem_filename (str): Path to the digital elevation model (DEM) file.
    - melt_magnitude (float): Magnitude of the meltwater.
    - x_center_of_melt (float): X-coordinate of the center of the melt region.
    - y_center_of_melt (float): Y-coordinate of the center of the melt region.
    - melt_width (float): Width of the melt region.
    - sparse (bool): Whether to use sparse arrays for the mlet and water_depth outputs (default: True).

    Returns:
    - results (xr.Dataset): Dataset containing the water depth, DEM, and melt data.

    """

    # Load the DEM
    dem = rioxarray.open_rasterio(dem_filename, chunks={})
    dem = dem.squeeze()

 
    melt, melt_filename, bounds = square_melt_region(dem, 
                                                        melt_magnitude, 
                                                        x_center_of_melt=x_center_of_melt, 
                                                        y_center_of_melt=y_center_of_melt, 
                                                        width=melt_width)  
    water_depth = fsm(dem_filename, melt_filename=melt_filename)        
    
    # name the xr.DataArrays
    water_depth.name = 'water_depth'
    dem.name = 'dem'
    melt.name = 'melt'

    # add information about the coordinates and variables in attributes
    melt.attrs = {'units': 'meters', 'long_name': 'surface melt', 'description': 'the surface melt as a function of x and y'}

    # merge the xr.DataArrays into a xr.Dataset
    results = xr.merge([water_depth, dem, melt])
    results = results.drop_vars('band')   # drop this unneeded variable

    bounds = np.array(bounds)
    results['bounds'] = xr.DataArray(bounds, dims=['bounds_index'], name='bounds')
    results.bounds.attrs = {'long_name': 'bounds of the rectangular melt region', 'description': 'the bounds of the rectangular melt region: (xmin, ymin, xmax, ymax)'}

    results = add_center_of_mass(results, 
                       x_center_of_melt, 
                       y_center_of_melt)
    
    results = GL_flux(results)
    
    if sparse:
        results = replace_dense_with_sparse(results, 'water_depth')
        results = replace_dense_with_sparse(results, 'melt')
    
    return results

def replace_dense_with_sparse(results: xr.Dataset, variable: str) -> xr.Dataset:
    """
    Replaces a dense variable in the results xr.DataSet with a sparse variable.

    Parameters:
        results (dict): The dictionary containing the results.
        variable (str): The name of the variable to be replaced.

    Returns:
        dict: The updated results dictionary with the variable replaced by a sparse variable.

    Notes:
        see BFRN_meltwater/python/notebooks/sparse.ipynb for notes on this. 
    """
    sparse_variable = sparse.COO.from_numpy(results[variable].values) 
    results[variable].values = sparse_variable
    return results


def load_REMA_subset(ROIgeojson_filename= '../../../ROIs/boudouin_west_1.geojson',
                     decimate=None,
                     coarsen=None,
                     rechunk=False,
                     save_tiff=False,
                     tiff_filename='/Users/jkingslake/Documents/science/meltwater_routing/big_data/REMA_subset.tif',
                     save_zarr=False,
                     zarr_filename='/Users/jkingslake/Documents/science/meltwater_routing/big_data/REMA_subset.zarr',
                     load=False,
                     previously_loaded_da=None):
    """
    Loads a subset of REMA (Reference Elevation Model of Antarctica) data based on a given region of interest (ROI).

    Parameters:
    - ROIgeojson_filename (str): Path to the ROI geojson file.
    - decimate (int): Decimation factor for subsampling the data. Default is None.
    - coarsen (int): Coarsening factor for aggregating the data. Default is None.
    - rechunk (bool): Flag indicating whether to rechunk the data. Default is False.
    - save_tiff (bool): Flag indicating whether to save the data as a TIFF file. Default is False.
    - tiff_filename (str): Path to save the TIFF file. Default is '/Users/jkingslake/Documents/science/meltwater_routing/big_data/REMA_subset.tif'.
    - save_zarr (bool): Flag indicating whether to save the data as a Zarr file. Default is False.
    - zarr_filename (str): Path to save the Zarr file. Default is '/Users/jkingslake/Documents/science/meltwater_routing/big_data/REMA_subset.zarr'.
    - load (bool): Flag indicating whether to load the data into memory. Default is False.
    - previously_loaded_da (xarray.DataArray): Previously loaded data array. Default is None.

    Returns:
    - da (xarray.DataArray): Subset of REMA data based on the ROI.
    - da_loaded (xarray.DataArray): Loaded data array if 'load' is True, otherwise None.
    """
    
    if previously_loaded_da is None:
        # get the crs by reading one file
        da_for_crs = rioxarray.open_rasterio("https://storage.googleapis.com/pangeo-pgc/8m/50_39/50_39_8m_dem_COG_LZW.tif", chunks={})

        # Load a geojson file created on geojson.io
        ROI = gpd.read_file(ROIgeojson_filename)
        ROI.to_crs(da_for_crs.spatial_ref.attrs['crs_wkt'],inplace=True)

        #read in the REMA tile index
        REMA_index = gpd.read_file('../../../../REMAWaterRouting/REMA_Tile_Index/REMA_Tile_Index_Rel1_1.shp')

        #get the bounds of the ROI
        [minx,miny,maxx,maxy]= ROI.bounds.values.tolist()[0]

        #create a shapely polygon from the bounds
        bbox = shapely.geometry.Polygon([[minx,miny],[maxx,miny],[maxx,maxy],[minx,maxy],[minx,miny]])

        #find the REMA tiles that intersect the ROI
        IS_intersection = np.argwhere(REMA_index.overlaps(bbox).tolist())

        #subset the list of REMA tiles to only include those that intersect the ROI
        IS_tiles = REMA_index.tile[IS_intersection.flatten()]

        #extract the row and column numbers of the REMA tiles from the subsetted list of tiles, IS_tiles
        row = [str.split(x,'_')[0] for x in IS_tiles.to_list()]
        col = [str.split(x,'_')[1] for x in IS_tiles.to_list()]
        row = np.int_(row)
        col = np.int_(col)

        ### Load the REMA tiles lazily
        uri_fmt = 'https://storage.googleapis.com/pangeo-pgc/8m/{i_idx:02d}_{j_idx:02d}/{i_idx:02d}_{j_idx:02d}_8m_dem_COG_LZW.tif'

        chunksize = 8 * 512
        rows = []
        for i in tqdm(np.arange(row.max(), row.min()-1, -1)): 
            cols = []
            for j in np.arange(col.min(),col.max()+1):
                uri = uri_fmt.format(i_idx=i, j_idx=j)
                try:
                    #print(uri)
                    dset = rioxarray.open_rasterio(uri, chunks=chunksize)
                    dset_masked = dset.where(dset > 0.0)
                    cols.append(dset_masked)
                    #print(uri)
                except RasterioIOError:
                    #print(f'failed to load tile {i},{j}')
                    pass
            rows.append(cols)

        dsets_rows = [xr.concat(row, 'x') for row in rows]
        da = xr.concat(dsets_rows, 'y', )
        da = da.squeeze()
        da = da.sel(x=slice(minx,maxx),y=slice(maxy,miny))

    else:
        da = previously_loaded_da

    da_loaded = None
    if load:
        da.load()
        da_loaded = da.copy()

    if decimate:
        da = da.isel(x=slice(0,da.x.shape[0],decimate),y=slice(0,da.y.shape[0],decimate))
        print(f"Resolution after decimation is {(da.x[1]-da.x[0]).values} m")
        da.load()


    if coarsen:
        da = da.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()
        print(f"Resolution after coarsening is {(da.x[1]-da.x[0]).values} m")
        da.load()        

    if rechunk:
        da = da.chunk({'x':chunksize,'y':chunksize})

    #add ocean cells around the edge of the DEM
    da[0:1,:] = 0.0 
    da[-1:,:] = 0.0
    da[:,0:1] = 0.0
    da[:,-1:] = 0.0

    #fill NaNs with zeros
    da = da.fillna(0.0)

    if save_tiff:
        da.rio.to_raster(tiff_filename)
    
    if save_zarr:
        da.to_zarr(zarr_filename)

    return da, da_loaded