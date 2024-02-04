% DEM = GRIDobj('Amery_40m.tif');  % this is the 40 m version that covers a larger part of the shelf % from REMA digital elevation model tiles mosaiced in ArcMap  (C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\BLCatchments.mxd).
% DEMc = crop(DEMc,'interactive');
% save ClippedDEMforCatchmentComparisons DEMc
load ClippedDEMforCatchmentComparisons DEMc
DEMc.Z(DEMc.Z<-1000) = nan;
imagesc(DEMc)

res = 100;
ii=1;
if res(ii)~=DEMc.cellsize
    DEMr = resample(DEMc,res(ii));
else
    DEMr = DEMc;
end

catchment_points = [1853980,720300; ...% large central lake ('Big Lake')
    1897820   ,   699780];


imagesc(DEM)

caxis([min(DEMr.Z(:)) max(DEMr.Z(:))])
caxis([min(DEMr.Z(:)) 1000])



%% 2 . isolate on catchment


 % 2a. Do the flow routing computations
        disp('Flow calculations..')
        
        FD = FLOWobj(DEMr);                % flow directions
%         A = flowacc(FD).*(FD.cellsize^2);  % flow accumulation
%         S   = STREAMobj(FD,A>1e7);         % stream locations(places where flow accumulation is larger than some threshold
        
        % compute drainage basins over the whole area
        [DB, x_outlet, y_outlet] = drainagebasins(FD,S);
        
        
 % 2b. Crop only the catchment we want and record is in a cell array (Mask) as a logical GridOBJ
        disp('isolating one catchment..')
        [x_b,y_b] = getcoordinates(DB);
        
        % find the correct catchment
        CatchmentNumber =  interp2(x_b,y_b,DB.Z,catchment_points(1,1),catchment_points(1,2),'nearest');  % (B.Z is an array of the catchment numbers) and these are the coordinates of the outlet of Big Lake's catchment
        
        Mask = DB == CatchmentNumber;   % mask is a logical array showing which cells are in our catchment.
        
        divides = Mask*nan;
        divides.Z = boundarymask(Mask.Z);
        divides.Z = imdilate(divides.Z,ones(2));
        hold on
        divides_plot = imagesc(divides);
        divides_plot.AlphaData = divides.Z;

        DEMBL = crop(DEMr,Mask,nan);   % make a new DEM which is just the the catchment we want.

        
       %% 
        figure 
        surf(DEMBL)
        DEMfs = fillsinks(DEMBL,10)
         
          figure 
          imagesc(DEMfs-DEMBL)
        
