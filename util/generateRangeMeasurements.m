function [rng_meas,rng_gt] = generateRangeMeasurements(pos_tag,pos_anchor,n_sigma)
  % Returns:
  %   input (2 x m matrix)
  %     each column represents represents one of the test inputs
  %   output (3 x m matrix)
  %     groundtruth at test points   
  n_anchor = size(pos_anchor,2);
  n_smp = size(pos_tag,2);
  rng_gt = zeros(n_anchor,n_smp);
  for j = 1:n_anchor
    rng_gt(j,:) = sqrt(sum((pos_tag - pos_anchor(:,j)).^2));
  end

  gauss = GaussianDist(zeros(n_anchor,1),diag(n_sigma^2*ones(1,n_anchor).^2));
  noise = gauss.sample(n_smp);
  rng_meas = 1.05*rng_gt + noise;
end
