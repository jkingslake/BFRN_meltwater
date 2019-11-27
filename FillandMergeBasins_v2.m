%% Script for filling all basins in a DEM with a catchment-area-dependent filling rate, and recomputing catchment areas as they fill. 
% clear all
% close all

plotWhenReComputeBasins=1; % dicates if we plot the new catchment each time we compute one.
DEMfilled = 0;

%% 1. Load DEM
% DEM = GRIDobj('C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\Downloaded_8m_tiles\Merged\Amery_40m.tif');  % this is the 40 m version that covers a larger part of the shelf % from REMA digital elevation model tiles mosaiced in ArcMap  (C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\BLCatchments.mxd).
% DEMc = crop(DEMc,'interactive');
% save ClippedDEMforPartialFillingAnalysis DEMc
% load ClippedDEMforCatchmentComparisons DEMc
load ClippedDEMforPartialFillingAnalysis DEMc
DEMc.Z(DEMc.Z<-1000) = nan;


%% 2. Decrease resolution. 
res =100;
ii=1;
if res(ii)~=DEMc.cellsize
    DEMr = resample(DEMc,res(ii));
else
    DEMr = DEMc;
end
cellArea = DEMr.cellsize^2;  
hs = fillsinks(DEMr,0.01);   % remove tiny basins
hs_original = hs;

%% 3. set up time domain
sec_in_day = 24*60*60;
m = 1/365;    % melt rate m^3/day/m^2 or m/day
dt = 0.01;      % days
T = 1000;     % days
t = 0:dt:T;  % days

%%. 4. Run Hypsometry_of_all_basins to compute the basins and their hypsometry
Hypsometry_of_all_basins

%% 5. Main loop over time domain
for ii =1:length(t)  % for each time step
    for jj = 2:length(BasinNumbers)   % for each basin
        if  b(jj).skip == 1  % skip this basin if there are no lakes in the catchment (should only be basins on the boundary.)
            continue
        end
        % plot a time series of lake level in every basin. 
%         if rem(ii,1)==0    
%             figure(222)
%             plot(t(ii),b(jj).h,'.')
%             hold on
%             drawnow
%             t(ii);
%         end
        
        
        % Lake surface area at this time step
        [~,NearestI] = min(abs(b(jj).h - b(jj).hw));
        A = NearestI*cellArea; % this works because the interp1 version was: A = interp1(b(kk).hw, [1:length(b(kk).hw)]'*cellArea , b(jj).h(ii));
        
        % Change the lake depth based on this area and the input rate: m*b(jj).BasinArea
        b(jj).h = b(jj).h + dt*m*b(jj).BasinArea/A;
          
    end
    
    % Catch if any basin is full
    if any([b.h]>=[b.maxdepth])
        h_old = hs;         % record the topography before updating it
        AddWaterDepthsTohs  % Use this script to update the
        Hypsometry_of_all_basins
        if DEMfilled == 1; return; end
    end
    disp([num2str(t(ii)) ' days'])
end



%%  Scrap


% % % imagesc(hs)
% % %             hold on
% % %             colormap flowcolor
% % %             caxis([0 3])
% % %             plot(x_s,y_s,'b','LineWidth',0.01)
% % %             drawnow

