classdef ParticleFilterCustom
  properties
    dist % (DiracDistribution) distribution of particles
  end
  
  methods
    function this = ParticleFilterCustom(dist_)
      % Constructor
      this.dist = dist_;
    end

    function this = setState(this,dist_)
      % Sets the current system state
      %
      % Parameters:
      %   distribution (AbstractHypertoroidalDistribution)
      %       new state
      % All automatic conversion can be done in the inherting classes
      assert(isa(dist_,class(this.dist)),'New distribution has to be of the same class as (or inhert from) the previous density.');
      this.dist = dist_;
    end
    
    function this = predict(this,noiseDistribution)
      % Predicts assuming identity system model, i.e.,
      % x(k+1) = x(k) + w(k),
      % where w(k) is additive noise given by noiseDistribution.
      %
      % Parameters:
      %   noiseDistribution (GaussianDistribution)
      %       distribution of additive noise
      noise = noiseDistribution.sample(numel(this.dist.w));
      this.dist.d = this.dist.d + noise;
    end
    
    function this = update(this,vlikelihood)
      % Updates assuming nonlinear measurement model
      % 
      % Parameters:
      %   vlikelihood (1 x n vector)
      this.dist = this.dist.reweigh(vlikelihood);

      % Resample
      this.dist.d = this.dist.sample(length(this.dist.d));
      % Reset weights to equal weights
      this.dist.w = 1/size(this.dist.d,2)*ones(1,size(this.dist.d,2));
    end
    
    function dist = getEstimate(this)
      % Return current estimate 
      %
      % Returns:
      %   dist (DiracDistribution)
      %       current estimate  
      dist = this.dist;
    end
  end
end

