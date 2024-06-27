
test_input = [cos(aoa_te);sin(aoa_te)];
n_tst = size(test_input,2);
[m_est_HvM,covar_HvM] = predictionMOGP(1,2*n_anchor,n_anchor,xeopt_hvM,{'hypertoroidalvMKernel'},training_points_unit,obs,test_input);  
m_est_grid_HvM = reshape(m_est_HvM,n_tst,n_anchor)';
sigma_HvM = reshape(sqrt(diag(covar_HvM)),n_tst,n_anchor)';

[m_est_SE,covar_SE] = predictionMOGP(n_anchor,1,n_anchor,xeopt_SE,{'SEKernel'},training_points,obs,aoa_te);  
m_est_grid_SE = reshape(m_est_SE,n_tst,n_anchor)';
sigma_SE = reshape(sqrt(diag(covar_SE)),n_tst,n_anchor)';

err_te = rng_meas_te - rng_gt_te;
idx_end = n_t;
idx_te = 1:2:idx_end;
lw = 2;

err_HvM = m_est_grid_HvM - rng_gt_te;
lb_HvM = err_HvM - 2*sigma_HvM;
ub_HvM = err_HvM + 2*sigma_HvM;

err_SE = m_est_grid_SE - rng_gt_te;
lb_SE = err_SE - 2*sigma_SE;
ub_SE = err_SE + 2*sigma_SE;

err_gauss = dist_rng_noise.mu.*ones(1,idx_end);
sigma_gauss = sqrt(diag(dist_rng_noise.C)).*ones(1,idx_end);
lb_gauss = err_gauss - 2*sigma_gauss;
ub_gauss = err_gauss + 2*sigma_gauss;

for i = 1:n_anchor-2
  plotErrorDist(idx_te,lb_HvM(i,idx_te),ub_HvM(i,idx_te),err_te(i,idx_te),...
    err_HvM(i,idx_te),lw);
  plotErrorDist(idx_te,lb_SE(i,idx_te),ub_SE(i,idx_te),err_te(i,idx_te),...
    err_SE(i,idx_te),lw);
  plotErrorDist(idx_te,lb_gauss(i,idx_te),ub_gauss(i,idx_te),err_te(i,idx_te),...
    err_gauss(i,idx_te),lw);
end