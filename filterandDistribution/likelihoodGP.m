function [pdf,valid] = likelihoodGP(pos_est,training_inputs,obs,pos_anchor,xeopt,rng_meas,kernel_type)
  % compute test AOA
  n_tst = size(pos_est,2);
  n_anchor = size(pos_anchor,2);
  aoa = zeros(n_anchor,n_tst);
  for i = 1:n_anchor
    aoa(i,:) = atan2(pos_anchor(2,i) - pos_est(2,:),pos_anchor(1,i) - pos_est(1,:));
  end
  % compute distriburion at test input
  if strcmp(kernel_type,'hypertoroidalvMKernel')
    test_input = [cos(aoa);sin(aoa)];
    [m_est,covar] = predictionMOGP(1,2*n_anchor,n_anchor,xeopt,kernel_type,training_inputs,obs,test_input);
  else
    test_input = aoa;
    [m_est,covar] = predictionMOGP(n_anchor,1,n_anchor,xeopt,kernel_type,training_inputs,obs,test_input);
  end
  m_est_grid = reshape(m_est,n_tst,n_anchor)';
  err = rng_meas - m_est_grid;
  if any(abs(err) > 3)
    valid = 0;
    pdf = 0;
    return
  end     
  % compute pdf
  pdf = zeros(1,n_tst);
  for i = 1:n_tst
    cov = covar(i:n_tst:end,i:n_tst:end);
    cov = 0.5*(cov + cov');
    gaussd = GaussianDist(zeros(n_anchor,1),cov);
    pdf(i) = gaussd.pdf(err(:,i));
  end
  valid = 1;
  if (~sum(pdf)>0) 
    valid = 0;
  end
end