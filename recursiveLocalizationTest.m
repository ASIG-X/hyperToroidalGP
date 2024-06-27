clear; close all;
cur_folder = fileparts(mfilename('fullpath'));
path_name = [cur_folder,'/eva/training_results/'];

n_sigmav = [0.01,0.03,0.05];
str_sigma = {'01','03','05'};
str_traj = {'lissa','lima','rho'};
n_t = 1000;

rec_width = 30;
rec_height = 30;

traj1 = genarateTrajectoryLissajous(pi/2,3,4,rec_width,rec_height,n_t);
traj2 = genarateTrajectoryLimacon(rec_width-8,rec_height-8,n_t) + [4;0];
traj3 = genarateTrajectoryRhodonea(3,rec_width-3.8,rec_height-3.8,n_t) + [2;-0.3];

fh_trajv = {traj1,traj2,traj3};

for n_level = 1:size(n_sigmav,2)
  n_sigma = n_sigmav(n_level);
  save_file_name = ['noise0_',str_sigma{n_level}];
  load([path_name,save_file_name,'/',save_file_name,'xe_hvM.mat']);
  load([path_name,save_file_name,'/',save_file_name,'xe_SE.mat']);
  load([path_name,save_file_name,'/',save_file_name,'traininginputs.mat']);
  load([path_name,save_file_name,'/',save_file_name,'traininginputs_unit.mat']);
  load([path_name,save_file_name,'/',save_file_name,'observations.mat']);
  load([path_name,save_file_name,'/',save_file_name,'noise_dist.mat']);
  load([path_name,save_file_name,'/',save_file_name,'pos_anchor.mat']);
  load([path_name,save_file_name,'/',save_file_name,'pos_tag_tr.mat']);  
    
  for n_traj = 1:size(str_traj,2)
    % generate trajectory
    pos_tag_te = fh_trajv{n_traj};
  
    mean_noise = zeros(2,1);
    cov_noise = diag([0.4,0.4].^2);
    dist_process_noise = GaussianDist(mean_noise,cov_noise);
    cov_noise = diag([1,1]);
    prior_noise = GaussianDist(mean_noise,cov_noise);

    [rng_meas_te] = generateRangeMeasurements(pos_tag_te,pos_anchor,n_sigma);
    aoa_te = computeAoAFromPosition(pos_tag_te,pos_anchor);
    n_te = size(pos_tag_te,2);
    % particle filter
    n_particles = 100;
  
    fh_lkh_HvM = @(x,y) likelihoodGP(x,training_points_unit,obs,pos_anchor,xeopt_hvM,y,{'hypertoroidalvMKernel'});
    pos_est_HvM = pf(rng_meas_te,dist_process_noise,prior_noise,fh_lkh_HvM,n_particles,pos_tag_te(:,1));
  
    fh_lkh_SE = @(x,y) likelihoodGP(x,training_points,obs,pos_anchor,xeopt_SE,y,{'SEKernel'});
    pos_est_SE = pf(rng_meas_te,dist_process_noise,prior_noise,fh_lkh_SE,n_particles,pos_tag_te(:,1));  

    fh_lkh_gauss = @(x,y) likelihoodGaussian(x,pos_anchor,y,dist_rng_noise);
    pos_est_gauss = pf(rng_meas_te,dist_process_noise,prior_noise,fh_lkh_gauss,n_particles,pos_tag_te(:,1));

    RMSE_hvM = sqrt(sum((pos_tag_te - pos_est_HvM).^2,'all')/n_te);
    RMSE_SE = sqrt(sum((pos_tag_te - pos_est_SE).^2,'all')/n_te);  
    RMSE_gauss = sqrt(sum((pos_tag_te - pos_est_gauss).^2,'all')/n_te);

    RMSE = [RMSE_hvM,RMSE_gauss,RMSE_SE];
    fprintf('str_trxaj: %s, noise level: %f\n',str_traj{ n_traj},n_sigmav(n_level));
    fprintf('RMSE: HvM - %.4f, Parametric - %.4f, SE - %.4f\n',RMSE);    
  end
end
