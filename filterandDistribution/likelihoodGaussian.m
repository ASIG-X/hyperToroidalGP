function [pdf,valid] = likelihoodGaussian(pos_est,pos_anchor,rng_meas,dist_rng_noise)
  % compute predicted range
  n_tst = size(pos_est,2);
  n_anchor = size(pos_anchor,2);
  rng_pred = zeros(n_anchor,n_tst);
  for i = 1:n_anchor
    rng_pred(i,:) = sqrt(sum((pos_est - pos_anchor(:,i)).^2,1));
  end
  % compute pdf
  err = rng_meas - rng_pred;
  if any(abs(err) > 3)
    valid = 0;
    pdf = 0;
    return
  end
  pdf = dist_rng_noise.pdf(err);
  valid = 1;
  if (~sum(pdf)>0) 
    valid = 0;
  end  
end