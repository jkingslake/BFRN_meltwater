# Python code calling fill-spill-merge

`download_REMA_subsets.ipynb` is for selecting, lazily loading, subsetting, downoading and writing to disk part of the Reference Elevation Model of Antarctic (REMA). 

`fsm_on_rema.ipynb` applies fill-spill-merge to small subsets of the REMA DEM. These subsets can be produced using `download_REMA_subsets.ipynb`. 

`simple_test_of_fsm.ipynb` tests out fill-spill-merge and our approach to calling it and loading the results using very simple DEMs that ae part of the test suite of part of the code. 

`centroid_test.ipynb` explores an xarray-based way of computing the center of mass of rasters, which is used in `fsm_on_rema.ipynb` and will be used elsewhere in the future. 

`fsm_results` stores the results of computations. 

`rema_subsets` stores subsets of the REMA DEM. 