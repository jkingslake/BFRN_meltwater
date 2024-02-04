% Take the water depths in each basin and fill the DEM up to that level.
% This script should be triggered every time a basin is filled, so the the
% hypsometry can be re-calculated. This should capture the new configuration, potentially with two basins now connected.

% method 1

for tt = 2:length(BasinNumbers)
    if  b(tt).skip == 1
        continue
    end
    Addition = min(b(tt).maxdepth,b(tt).h);
    hs.Z(b(tt).MaskI) = max(h_old.Z(b(tt).MaskI) , min(h_old.Z(b(tt).MaskI)) + Addition );
    
    % add a percentage of filled-unfilled equal to h/max(depth)
end

% method 2

% for kk = find(BasinNumbers==17);%1:length(BasinNumbers)
%      fractionFilled = hs*0;
%      fractionFilled.Z(b(kk).MaskI) =min(1,(b(kk).h(ii+1))/b(kk).maxdepth); % limit to 1
% end
% % update the DEM with the correct level of water in each basin, using the
% % proportion of the filled value. The advantage of doing it this way is
% % that it should maintain the slight surface slope that fillsinks includes
% % to allow routing across the flat areas.
%
% hs = h_old + fractionFilled.*P_all;

