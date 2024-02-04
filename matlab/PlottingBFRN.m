



Loading_BFRN


histogram((B.Z(:),-0.01:0.0001:0.01)

imagesc(B)

ca= caxis;
caxis([0 0.05])
ax = axis;
[xg,yg] = antbounds_data('gl','xy');
hold on
plot(xg,yg,'r')

axis(ax)

