clear; close all;
cur_folder = fileparts(mfilename('fullpath'));
path_name = [cur_folder,'/eva/training_results/'];

n_sigmav = [0.01,0.03,0.05];
str_sigma = {'01','03','05'};

% simulation
rec_width = 30;
rec_height = 30;
n_anchor = 3;
pos_anchor = zeros(2,n_anchor);
pos_anchor(:,1) = [rec_width/2;rec_height/2 + sqrt(3)*rec_width/8];
pos_anchor(:,2) = [rec_width/4;rec_height/2 - sqrt(3)*rec_width/8];
pos_anchor(:,3) = [3*rec_width/4;rec_height/2 - sqrt(3)*rec_width/8];

n_smp_grid = [30;24];
pos_tag_x = linspace(0,rec_width,n_smp_grid(1));
pos_tag_y = linspace(0,rec_height,n_smp_grid(2));
[pos_grid_x,pos_grid_y] = meshgrid(pos_tag_x,pos_tag_y);
pos_tag_tr = [pos_grid_x(:),pos_grid_y(:)]'; 
pos_tag_tr = [pos_tag_tr, pos_anchor(:,1) - [1.5;0.5], ...
  pos_anchor(:,1) + [1.5;0.5], pos_anchor(:,1) - [1.5;-0.5], ...
  pos_anchor(:,1) + [1.5;-0.5]];
n_data = length(pos_tag_x)*length(pos_tag_y);
idx_smp = 1:3:n_data;
idx_smp = [idx_smp,n_data+1:n_data+4];
n_smp = size(idx_smp,2);
for n_level = 1:size(n_sigmav,2)
  n_sigma = n_sigmav(n_level);
  save_file_name = ['noise0_',str_sigma{n_level}];
  dir_name = [path_name,save_file_name,'/'];
  
  if ~isfolder(dir_name)
    mkdir(dir_name);
  end
  
  [rng_meas,rng_gt] = generateRangeMeasurements(pos_tag_tr,pos_anchor,n_sigma);
  
  aoa_sum = computeAoAFromPosition(pos_tag_tr,pos_anchor);

  obs = reshape(rng_meas(:,idx_smp)',1,n_smp*n_anchor);
  
  % test data
  idx_te = setdiff(1:n_data,idx_smp);
  aoa_te = aoa_sum(:,idx_te);
  rng_gt_te = rng_gt(:,idx_te);
  n_te = size(idx_te,2);
  pos_tag_te = pos_tag_tr(:,idx_te);

  % figure
  % scatter(pos_anchor(1,:),pos_anchor(2,:),'r','filled')
  % hold on 
  % scatter(pos_tag_tr(1,idx_smp),pos_tag_tr(2,idx_smp),'b','filled','MarkerFaceAlpha',0.5)
  % scatter(pos_tag_te(1,:),pos_tag_te(2,:),'*c')
  % legend('anchor','training set','test set')  
  % axis equal

  % Estimation Settings
  num_kernel = 1;
  out_dim = n_anchor;
  in_dim = n_anchor;
  % initial hyperparameters for MOGP using different kernels
  B = eye(out_dim);
  B(B<0.01) = 0.01;
  param_MOGP = B(:)'; % parameters for the LCM kernel in multioutput GP
  param_noise = 0.5*ones(1,out_dim); % parameters for noise
  
  % Estimation MOGP hvM
  training_points_unit = [cos(aoa_sum(:,idx_smp));sin(aoa_sum(:,idx_smp))];
  x0 = [[1,ones(1,in_dim),ones(1,in_dim)],param_MOGP,param_noise];
  [xeopt_hvM] = optimizationMOGP(num_kernel,2*in_dim,out_dim,x0,{'hypertoroidalvMKernel'},...
    training_points_unit,obs);

  % SE
  x0 = [[1,ones(1,in_dim)],param_MOGP,param_noise];
  training_points = aoa_sum(:,idx_smp);
  [xeopt_SE] = optimizationMOGP(n_anchor,1,out_dim,x0,{'SEKernel'},training_points,obs);
  
  % estimate Gaussian distribution
  dist_rng_noise = estGaussian(rng_meas(:,idx_smp),rng_gt(:,idx_smp));
  
  save([dir_name,save_file_name,'observations.mat'],'obs');
  save([dir_name,save_file_name,'traininginputs.mat'],'training_points');
  save([dir_name,save_file_name,'traininginputs_unit.mat'],'training_points_unit');
  save([dir_name,save_file_name,'xe_hvM.mat'],'xeopt_hvM');
  save([dir_name,save_file_name,'xe_SE.mat'],'xeopt_SE');
  save([dir_name,save_file_name,'noise_dist.mat'],'dist_rng_noise');
  save([dir_name,save_file_name,'pos_anchor.mat'],'pos_anchor');
  save([dir_name,save_file_name,'pos_tag_tr.mat'],'pos_tag_tr');
end