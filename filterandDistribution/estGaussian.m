function gauss = estGaussian(rng_meas,rng_gt)
  n_anchor = size(rng_meas,1);
  param = zeros(2,n_anchor);
  for j = 1:n_anchor
    err = rng_meas(j,:) - rng_gt(j,:);
    pd = fitdist(err','Normal');
    param(:,j) = [pd.mu;pd.sigma];
%     figure
%     pd.plot;
%     fprintf('mean error: %.4f\n',mean(err(~isnan(err))));
  end  
  gauss = GaussianDist(param(1,:),diag(param(2,:).^2));
end