% Script for testing out a way of incrementally filling a sinlge basin. 
% J. Kingslake 11-19-19


% DEM = GRIDobj('C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\Downloaded_8m_tiles\Merged\Amery_40m.tif');  % this is the 40 m version that covers a larger part of the shelf % from REMA digital elevation model tiles mosaiced in ArcMap  (C:\Users\jkingslake\Documents\Remote Sensing\Antarctica Whole\REMA\BLCatchments.mxd).
% DEMc = crop(DEM,'interactive');
% save ClippedDEMforPartialFillingAnalysis DEMc
load ClippedDEMforPartialFillingAnalysis DEMc
DEMc.Z(DEMc.Z<-1000) = nan;


res = 50;
ii=1;
if res(ii)~=DEMc.cellsize
    DEMr = resample(DEMc,res(ii));
else
    DEMr = DEMc;
end


%% 2a. subset
disp('Flow calculations..')

FD = FLOWobj(DEMr,'preprocess','none','verbose',true,'mex',true);                % flow directions

% compute drainage basins over the whole area
[DB, x_outlet, y_outlet] = drainagebasins(FD);
[x_b,y_b] = getcoordinates(DB);


point = [1848420    625740];   %The location of the large lake
% find the correct catchment
CatchmentNumber =  interp2(x_b,y_b,DB.Z,point(1),point(2),'nearest');  % (B.Z is an array of the catchment numbers) and these are the coordinates of the outlet of Big Lake's catchment

Mask = DB == CatchmentNumber;   % mask is a logical array showing which cells are in our catchment.

DEMsubcatch = crop(DEMr,Mask,nan);   % make a new DEM which is just the the catchment we want.

figure
subplot 131
imagesc(DB); colormap lines
subplot 132
imagesc(Mask); colormap parula
subplot 133
imagesc(DEMsubcatch)


%%
Filled = fillsinks(DEMsubcatch);
P = Filled-DEMsubcatch;

FDsub = FLOWobj(DEMsubcatch,'preprocess','none','verbose',true,'mex',true);                % flow directions
A = flowacc(FDsub).*(FDsub.cellsize^2);  % flow accumulation
S   = STREAMobj(FDsub,A>1e5);         % stream locations(places where flow accumulation is larger than some threshold

% compute drainage basins over the whole area
[DB_sub, x_outlet, y_outlet] = drainagebasins(FDsub,S);



%% 3. plot computed ponds, plot streams, catchments and label catchments
figure
imagesc(P)
hold on
colormap flowcolor
caxis([0 7])
% set(gcf,'Pos',[2561   401  1920   963])
set(gcf,'Pos', [573   438   560   420])
d = DB_sub;
d.Z = boundarymask(DB_sub.Z);
% d.Z = imdilate(d.Z,ones(5));
% d.Z = imdilate(d.Z,ones(5));
ax = axis;
d_plot = imagesc(d);
d_plot.AlphaData = d.Z;
axis(ax)

[x_s,y_s] = STREAMobj2XY(S);
hold on
plot(x_s,y_s,'b','LineWidth',0.01)


%% hist of P

depths = P.Z(~isnan(P.Z) & P.Z~=0);

heights = max(depths) - depths;
heights_sorted = sort(heights);

figure
bar(heights_sorted)
% histogram(heights_sorted)
A_h = (1:length(heights_sorted) )*P.cellsize^2;
h=heights_sorted;
D = 2*sqrt(A_h/pi)
plotyy(h,A_h,h,D)
%%
DEMsubcatch_new = DEMsubcatch;
sec_in_day = 24*60*60;
Qin = 1*sec_in_day;   % m^3 day^-1
dt = 0.01;      % days
T = 100;     % days
t = 0:dt:T;  % days
% solve dH/dt = A(h) dh/dt = Qin;  V(0) = 0; h(0) = 0;
h = 0;
for ii =1:length(t)
    A = interp1(heights_sorted, A_h,h);
    h = h + dt*(Qin/A)
    
    DEMsubcatch_new.Z = max(DEMsubcatch.Z,min(DEMsubcatch.Z,[],'all')+h);
    
    if rem(ii,100)==0
        imagesc(DEMsubcatch_new-DEMsubcatch)
        hold on
        colormap flowcolor
        caxis([0 3])
        plot(x_s,y_s,'b','LineWidth',0.01)
        drawnow
        t(ii)
    end
    
    
end





