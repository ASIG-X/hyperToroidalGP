function pos = genarateTrajectoryLimacon(rec_width,rec_height,n_t)
  phi = linspace(0,2*pi,n_t);
  r = 0.28*rec_width*(1 + 2*cos(phi));
  [x,y] = pol2cart(phi,r);
  pos = [x;y];  
  theta = -pi/2;
  R = [cos(theta),-sin(theta);
       sin(theta),cos(theta)];
  pos = R*pos;  
  pos = pos + [rec_width/2;rec_height/2+11];    
end