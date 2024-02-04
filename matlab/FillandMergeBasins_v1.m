%% Script for filling all basins in a DEM with am cartchment-area-dependent filling rate.


% DEM = GRIDobj('C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\Downloaded_8m_tiles\Merged\Amery_40m.tif');  % this is the 40 m version that covers a larger part of the shelf % from REMA digital elevation model tiles mosaiced in ArcMap  (C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\BLCatchments.mxd).
% DEMc = crop(DEM,'interactive');
% save ClippedDEMforPartialFillingAnalysis DEMc
load ClippedDEMforPartialFillingAnalysis DEMc
DEMc.Z(DEMc.Z<-1000) = nan;

plotWhenReComputeBasins=1;
res =100;
ii=1;
if res(ii)~=DEMc.cellsize
    DEMr = resample(DEMc,res(ii));
else
    DEMr = DEMc;
end
cellArea = DEMr.cellsize^2;
hs = fillsinks(DEMr,0.01);   % remove tiny basins


%%
sec_in_day = 24*60*60;
m = 1/365;    % melt rate m^3/day/m^2 or m/day
dt = 0.01;      % days
T = 1000;     % days
t = 0:dt:T;  % days

Hypsometry_of_all_basins


for ii =1:length(t)   
    for jj = find(BasinNumbers==17);%2:length(BasinNumbers)        
        if rem(ii,100)==0
            figure(222)
            plot(t(ii),b(jj).h(ii),'.')
            hold on
            drawnow
            t(ii);
        end
        
        
        % area at this time step
        [~,NearestI] = min(abs(b(jj).h(ii) - b(kk).hw));
        A = NearestI*cellArea; % this works because the interp1 version was: A = interp1(b(kk).hw, [1:length(b(kk).hw)]'*cellArea , b(jj).h(ii));
        b(jj).h(ii+1) = b(jj).h(ii) + dt*m*b(jj).BasinArea/A;
               
        %%%%% catch if the basin is full        
        if b(jj).h(ii+1)>=max(b(jj).hw)
            h_old = hs;
            AddWaterDepthsTohs
            Hypsometry_of_all_basins
        end
        
     
    end  
end



% % % imagesc(hs)
% % %             hold on
% % %             colormap flowcolor
% % %             caxis([0 3])
% % %             plot(x_s,y_s,'b','LineWidth',0.01)
% % %             drawnow

