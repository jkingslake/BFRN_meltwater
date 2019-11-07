%% Investigate the impact of resolution on  catchment area calculations in two catchments on the Amery Ice Shelf
% make comparisons in several places
% start with Amery


%% Load the DEM

% close all
% clear all
DEM = GRIDobj('C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\Downloaded_8m_tiles\Merged\Amery_40m.tif');  % this is the 40 m version that covers a larger part of the shelf % from REMA digital elevation model tiles mosaiced in ArcMap  (C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\BLCatchments.mxd).
% DEMc = crop(DEM,'interactive');
% save ClippedDEMforCatchmentComparisons DEMc
load ClippedDEMforCatchmentComparisons DEMc
DEMc.Z(DEMc.Z<-1000) = nan;

res = [50:10:200];

catchment_points = [1853980,720300; ...% large central lake ('Big Lake')
     1897820   ,   699780];   

for kk = 1:size(catchment_points,1)   % loop over the catchments
    for ii = 1:length(res)            % loop over the resolutions. 
        disp(['starting ' num2str(res(ii)) ' m' ])
        disp('Resampling..')

        %% 1. For different resolutions, resample, do the catchment computation
        if res(ii)~=DEMc.cellsize
            DEMr = resample(DEMc,res(ii));
        else
            DEMr = DEMc;
        end
        

        
        %% 2. Do the flow routing computations
        disp('Flow calculations..')
        
        FD = FLOWobj(DEMr);                % flow directions
        A = flowacc(FD).*(FD.cellsize^2);  % flow accumulation
        S   = STREAMobj(FD,A>1e7);         % stream locations(places where flow accumulation is larger than some threshold
        
        % compute drainage basins over the whole area
        [DB, x_outlet, y_outlet] = drainagebasins(FD,S);
        
        
        %% 3. Crop only the catchment we want and record is in a cell array (Mask) as a logical GridOBJ         
        disp('isolating one catchment..')
        [x_b,y_b] = getcoordinates(DB);
        
        % find the correct catchment 
        CatchmentNumber =  interp2(x_b,y_b,DB.Z,catchment_points(kk,1),catchment_points(kk,2),'nearest');  % (B.Z is an array of the catchment numbers) and these are the coordinates of the outlet of Big Lake's catchment
        
        Mask{ii,kk} = DB == CatchmentNumber;   % mask is a logical array showing which cells are in our catchment.
   
        
        % compute drainage divides
%         divides = DB*nan;
%         divides.Z = boundarymask(DB.Z);
%         divides.Z = imdilate(divides.Z,ones(2));
% 
%         [x_catch_mesh,y_catch_mesh] = meshgrid(x_b,y_b);
%         I = Mask{ii,kk}.Z(:)==1;
%         X_catch = x_catch_mesh(I);
%         Y_catch = y_catch_mesh(I);
%         
%         imagesc(Mask{ii,kk})
%         figure         
%         plot(X_catch,Y_catch,'.')
%         hold on
%         
        disp(['finished ' num2str(res(ii)) ' m' ])
        
    end
    kk
end


%% Plot catchments in map view and plot their total areas (in m^2) for each resolution and catchment 

figure(2)
kk=1
hd = tight_subplot(size(Mask,1), 2, 0)
for ii=1:size(Mask,1)
    figure(1)
    subplot(121)
    plot(Mask{ii,kk}.cellsize,nnz(Mask{ii,kk}.Z(:)==1)*Mask{ii,kk}.cellsize.^2,'+b')
    hold on
    
    %%%
    [x_catch_vec,y_catch_vec] = getcoordinates(Mask{ii,kk});
    [x_catch_mesh,y_catch_mesh] = meshgrid(x_catch_vec,y_catch_vec);
    X_catch = x_catch_mesh(Mask{ii,kk}.Z(:)==1);
    Y_catch = y_catch_mesh(Mask{ii,kk}.Z(:)==1);
    
    figure(2)     
    axes(hd(ii*2-1))
    plot(X_catch,Y_catch,'.')
    axis('off')
    text(mean(xlim),mean(ylim),num2str(Mask{ii,kk}.cellsize))
    ii    
end
figure(1)
subplot(121)
xlabel  'cellsize [m]'
ylabel 'catchment size [m^2]'

figure(2)
kk=2
for ii=1:size(Mask,1)
    figure(1)
    subplot(122)    
    plot(Mask{ii,kk}.cellsize,nnz(Mask{ii,kk}.Z(:)==1)*Mask{ii,kk}.cellsize.^2,'+b')
    hold on

    %%%
    [x_catch_vec,y_catch_vec] = getcoordinates(Mask{ii,kk});
    [x_catch_mesh,y_catch_mesh] = meshgrid(x_catch_vec,y_catch_vec);
    X_catch = x_catch_mesh(Mask{ii,kk}.Z(:)==1);
    Y_catch = y_catch_mesh(Mask{ii,kk}.Z(:)==1);
    
    figure(2)
    axes(hd(ii*2))
    plot(X_catch,Y_catch,'.')
    axis('off')
    text(mean(xlim),mean(ylim),num2str(Mask{ii,kk}.cellsize))
    ii
end

figure(1)
subplot(122)    
xlabel  'cellsize [m]'
ylabel 'catchment size [m^2]'

figure(1)
print('Total_Area_cellsize.png','-dpng')


figure(2)
% print('Amery_Catchments.png','-dpng')



% % % % %% 6. plot computed ponds, plot streams, catchments and label catchments
% % % % figure
% % % % imagesc(P)
% % % % colormap flowcolor
% % % % caxis([0 1])
% % % % % imagesc(DB)
% % % % hold on
% % % % [x_s,y_s] = STREAMobj2XY(S);
% % % % plot(x_s,y_s,'b','LineWidth',0.01)
% % % % 
% % % % divides.Z = imdilate(divides.Z,ones(5));
% % % % divides_plot = imagesc(divides);
% % % % divides_plot.AlphaData = divides.Z;
% % % % 
% % % % 
% % % % ax = axis;
% % % % hold on
% % % % [xg,yg] = antbounds_data('gl','xy');
% % % % plot(xg,yg,'r')
% % % % axis(ax)
% % % % set(gcf,'Pos',[ 110         462        2068         787])
% % % % axis(ax)
% % % % 
% % % % 
% % % % 
% % % % 
% % % % % %% 5. Save
% % % % % savename = ['AmeryMask_' num2str(res)];
% % % % % eval([savename '= AmeryMask'])
% % % % % save(savename, savename)
