function pos = genarateTrajectoryRhodonea(c,rec_width,rec_height,n_t)
  phi = linspace(0,2*pi,n_t);
  r = 0.5*rec_width*sin(c*phi);
  [x,y] = pol2cart(phi,r);
  pos = [x;y];
  theta = pi/3;
  R = [cos(theta),-sin(theta);
     sin(theta),cos(theta)];
  pos = R*pos;
  % pos = rotateTraj(pi/3,pos);
  pos = pos + [rec_width/2;rec_height/2];
end

% function pos = getRhodonea(c,rec_width,rec_height,n_t)
%   phi = linspace(0,2*pi,n_t);
%   r = 0.5*rec_width*sin(c*phi);
%   [x,y] = pol2cart(phi,r);
%   pos = [x + rec_width/2;
%          y + rec_height/2];
% end

% function pos = rotateTraj(theta,pos)
%   R = [cos(theta),-sin(theta);
%        sin(theta),cos(theta)];
%   pos = R*pos;
% end