%% Script for filling all basins in a DEM with a catchment-area-dependent filling rate, and recomputing catchment areas as they fill.
clear all
% close all

plotWhenReComputeBasins=1; % dicates if we plot the new catchment each time we compute one.
DEMfilled = 0;
%% Idealized surface

% Fully filled troughs, no annular lakes
%     idealized = 300*ones(1000,10000);
%     x = 0:0.0001:1;
%     slope = repmat(-200*x,1000,1);
%     x = 0:pi/500:2*pi;
%     acrossflow = repmat(20*sin(10*x)',1,10000);
%     x = 0:pi/5000:2*pi';
%     alongflow = repmat(10*sin(5*x),1000,1);
%     idealized = idealized+slope(:,2:end)+alongflow(:,2:end)+acrossflow(2:end,:);
%     idealized(:,1) = 350;

idealized = 300*ones(1000,10000);
x = 0:0.0001:1;
slope = repmat(-100*x,1000,1);
x = 0:pi/500:2*pi;
acrossflow = repmat(5*sin(5*x+3.14/2)',1,10000);
x = 0:pi/5000:2*pi';
alongflow = repmat(6*sin(10*x),1000,1);
idealized = idealized+slope(:,2:end)+alongflow(:,2:end)+acrossflow(2:end,:);
idealized(:,1)=320;

[X,Y] = meshgrid(1:1:10000,1:1:1000);
hs = GRIDobj(X,Y,idealized);
hs_original = hs;
cellArea = 1;
DEMfilled = 0;


% 3. set up time domain
sec_in_day = 24*60*60;
%m = 1/365;    % melt rate m^3/day/m^2 or m/day
dt = 0.03;      % days
T = 90;     % days
t = 0:dt:T;  % days

summer_count=1;
%%. 4. Run Hypsometry_of_all_basins to compute the basins and their hypsometry
Hypsometry_of_all_basins
water = hs;
water.Z = zeros(hs.size);
figure(1)
clf
plot(hs_original.Z(500,:),'k')
%% 5. Main loop over time domain
while summer_count<3
    plotWhenReComputeBasins=0;
    Hypsometry_of_all_basins
    for ii =1:length(t)  % for each time step
        
        for jj = 1:length(BasinNumbers)   % for each basin
            if  b(jj).skip == 1  % skip this basin if there are no lakes in the catchment (should only be basins on the boundary.)
                continue
            end
            k = find(P_all.Z(500,:),1,'first');
            northernedge = unique(DB.Z(:,k));
            m = zeros(length(BasinNumbers),1);
            m(northernedge) = .1/365;
            % Lake surface area at this time step
            [~,NearestI] = min(abs(b(jj).h - b(jj).hw));
            Area = NearestI*cellArea; % this works because the interp1 version was: A = interp1(b(kk).hw, [1:length(b(kk).hw)]'*cellArea , b(jj).h(ii));
            % Change the lake depth based on this area and the input rate: m*b(jj).BasinArea
            b(jj).h = b(jj).h + dt*m(jj)*b(jj).BasinArea/Area;
         
        end
         plot([b(:).h]); drawnow
        % Catch if any basin is full
        if any([b.h]>=[b.maxdepth])
            filledbasins = find([b.h]>=[b.maxdepth]);
            for s=1:length(filledbasins)
                disp(['Basin ', num2str(b(filledbasins(s)).BasinNumber),' has been filled'])
            end
            h_old = hs;         % record the topography before updating it
            AddWaterDepthsTohs  % Use this script to update the
            %incision %incise 3 pixels outwards from water
            hs.Z(:,1)=320;
            plotWhenReComputeBasins=1;
            Hypsometry_of_all_basins
            if DEMfilled == 1; return; end
        end
        disp([num2str(t(ii)) ' days'])
        
        
        
    end
    
    summer_count = summer_count+1;
    % %Post-Summer Freezing
    for tt = 1:length(BasinNumbers)
        if  b(tt).skip == 1
            continue
        end
        [~,NearestI] = min(abs(b(tt).h - b(tt).hw));
        depths = P_all.Z(b(tt).MaskI);
        [~, sortedIndex] = sort(depths);
        SubMask = sortedIndex(1:NearestI);
        Addition = b(tt).hw(NearestI);
        
        %MaskI =find(flipud(b(tt).MaskLogical.Z));
        hs.Z(b(tt).MaskI(SubMask)) = min(hs.Z(b(tt).MaskI(SubMask))) + Addition;
        %prevent 'negative water' by replacing the areas where highpoints are removed.
        %negative_water = (hs_original.Z-hs.Z)>0;
        %hs.Z(negative_water) = hs_original.Z(negative_water);
        
        water = hs.Z-hs_original.Z;
        
        
        % add a percentage of filled-unfilled equal to h/max(depth)
    end
    end_of_summer_melt_refrozen = (water)*0.11; %slight expansion of refrozen melt
    hs = hs + end_of_summer_melt_refrozen;
    figure(1); hold on; plot(hs.Z(500,:),'b')
end
