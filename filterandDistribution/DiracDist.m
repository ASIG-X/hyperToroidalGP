classdef DiracDist
  properties
    d (:,:) double % Dirac locations
    w (1,:) double % Weights
  end
  
  methods
    function this = DiracDist(d_, w_)
      % Constructor, w_ is optional
      %
      % Parameters:
      %   d_ (dim x L)
      %       Dirac locations
      %   w_ (1 x L)
      %       weights for each Dirac
      arguments
          d_ (:,:) double {mustBeNonempty}
          w_ (1,:) double = ones(1,size(d_,2))/size(d_,2);
      end
      if size(d_,1)>size(d_,2)
          warning('Not even one one Dirac per dimension. If this warning is unxpected, verify d_ is shaped correctly.');
      end
      assert(size(d_,2) == size(w_,2),'Number of Dircas and weights must match.');
      this.d = d_;
      if ~(abs(sum(w_)-1)<1e-10)
          warning('Dirac:WeightsUnnormalized','Sum of the weights is not 1. Normalizing to 1')
          this.w = w_/sum(w_);
      else
          this.w = w_;
      end
    end
    
    function dist = reweigh(this,wNew)
      % Calculate the weight of each Dirac component. 
      % The new weight is given by the product of the old weight and the 
      % weight obtained with the 
      %
      % Parameters:
      %   wNew (1 x n vector)
      %     likelihood of the measurement given the current distribution
      % Returns:
      %   wd (some DiracDistribution)
      %       distribution with new weights and same Dirac locations
      assert(isequal(size(wNew),[1,size(this.d,2)]),'Function returned wrong number of outputs.');
      assert(all(wNew >= 0));
      assert(sum(wNew) > 0);
      
      dist = this;
      dist.w = wNew.*this.w;
      dist.w = dist.w/sum(dist.w);
    end

    function s = sample(this, n)
      % Obtain n samples from the distribution
      %
      % Parameters:
      %   n (scalar)
      %       number of samples
      % Returns:
      %   s (dim x n)
      %       one sample per column
      ids = discreteSample(this.w,n);
      s = this.d(:,ids);
    end

    function s = plotdist(this)
      s = scatter(this.d(1,:),this.d(2,:),this.w*20*size(this.w,2));
    end

    function E = expectation(this)
      E = sum(this.d.*this.w,2);
    end    
  end
end

