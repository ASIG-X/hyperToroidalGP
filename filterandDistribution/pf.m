function pos_est = pf(rng_meas_te,dist_process_noise,prior_noise,fh_lkh,n_particles,pos_init)
  % initialization
  init_state = pos_init.*ones(2,n_particles) + prior_noise.sample(n_particles);
  state_dist = DiracDist(init_state,1/n_particles*ones(1,n_particles));
  pf = ParticleFilterCustom(state_dist);
  n_meas = size(rng_meas_te,2);
  pos_est = zeros(2,n_meas);
  
  % estimation
  k = 1;
  state_dist = pf.getEstimate();
  pos_particles = [state_dist.d];

  vlik = fh_lkh(pos_particles,rng_meas_te(:,k));
  pf = pf.update(vlik);
  state_dist = pf.getEstimate();
  pos_est(:,k) = state_dist.expectation();
  
  for k = 2:n_meas
    % prediction
    pf = pf.predict(dist_process_noise);
    state_dist = pf.getEstimate();
    
    % compute likelihood
    pos_particles = state_dist.d;
    [vlik,valid] = fh_lkh(pos_particles,rng_meas_te(:,k));
    if ~valid
      pos_est(:,k) = state_dist.expectation();
      continue;
    end
    % update
    pf = pf.update(vlik);
    
    state_dist = pf.getEstimate();
    pos_est(:,k) = state_dist.expectation();
  end
end