function pos = genarateTrajectoryLissajous(delta,a,b,rec_width,rec_height,n_t)
  t = linspace(0,2*pi,n_t);
  x = 0.4*rec_width*sin(a*t + delta);
  y = 0.4*rec_width*sin(b*t);
  pos = [x + rec_width/2;
         y + rec_height/2];
end