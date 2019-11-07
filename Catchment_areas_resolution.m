%% investigate the impact of resolution on the catchment area calculations. 
% make comparisons in several places
% start with Amery


%% 1. Load the DEM
   
% close all
% clear all
DEM = GRIDobj('C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\Downloaded_8m_tiles\Merged\Amery_40m.tif');  % this is the 40 m version that covers a larger part of the shelf % from REMA digital elevation model tiles mosaiced in ArcMap 
% (C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\BLCatchments.mxd).


DEM.Z(DEM.Z<-1000) = nan;

%% 2. (optionally) Resample, do the basin-filling computation and save a Geotiff
res = 40;
if res~=DEM.cellsize
    DEMr = resample(DEM,res);
else 
    DEMr = DEM;
end


%% 3. Do the flow routing computations and (optionally) save the drainage divides as a raster. 
FD = FLOWobj(DEMr);
A = flowacc(FD).*(FD.cellsize^2);
S   = STREAMobj(FD,A>1e7);

% compute drainage basins over the whole area
[DB, x_outlet, y_outlet] = drainagebasins(FD,S);


%% 4. Crop only the catchment we want and save it as a logical GridOBJ
% find the correct catchment (the one with the Big Lake inlet in it). 
[x_b,y_b] = getcoordinates(DB);
CatchmentNumber =  interp2(x_b,y_b,DB.Z,1853980,720300,'nearest');  % (B.Z is an array of the catchment numbers) and these are the coordinates of the outlet of Big Lake's catchment

AmeryMask = DB == CatchmentNumber;   % mask is a logical array showing which cells are in our catchment. 

%% 5. Save
savename = ['AmeryMask_' num2str(res)];
eval([savename '= AmeryMask'])
save(savename, savename)
