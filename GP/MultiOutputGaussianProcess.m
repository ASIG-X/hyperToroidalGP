classdef MultiOutputGaussianProcess
  properties
    cov % kernel function (function handle)
    dimY % dimension of output y (integer)
  end

  methods
    function GP = MultiOutputGaussianProcess(cov_,dimY_)
      % Constructor
      %
      % Parameters:
      %   cov_ (function handle)
      %      kernel function
      %   dimY_ (int)
      %      dimension of output
      GP.cov = cov_;
      GP.dimY = dimY_;
    end

    function [mu,covm] = predict(this,x,y,sigma,xt)
      % Predictive distributions for multioutput Gaussian process regression
      % Parameters:
      %   x (D x n matrix)
      %     each column represents one of the n training points 
      %   y (1 x n*this.dimY vector)
      %     observations at training points x
      %   sigma (this.dimY x 1 vector)
      %     standard deviation of the observation noise in each dimension
      %   xt (D x m matrix)
      %     each column represents one of the m test points in R^d 
      % Returns:
      %   mu (m*this.dimY x 1 vector)
      %     predictive means at test points xt
      %   covm (m*this.dimY x m*this.dimY matrix)
      %     covariance matrices of the joint distribution at test points xt
      numx = size(x,2);
      numxt = size(xt,2);
      Ky = this.cov(x,x) + diag(reshape(repmat(sigma.^2,1,numx)',1,this.dimY*numx));
      L = chol(Ky);
      alpha = L\(L'\y');        
      Ks = this.cov(x, xt);
      KsT = Ks';
      mu = KsT*alpha;
      covm = this.cov(xt, xt) - (KsT)*(Ky\Ks) + ...
        diag(reshape(repmat(sigma.^2,1,numxt)',1,this.dimY*numxt));
    end

    function [varargout] = computeLogMarginalLik(this, x, y, sigma)
      % Log marginal likelihood and derivatives for multioutput Gaussian process regression
      % Parameters:
      %   x (D x n matrix)
      %     each column represents one of the n training points 
      %   y (1 x n*this.dimY vector)
      %     observations at training points x
      %   sigma (this.dimY x 1 vector)
      %     standard deviation of the observation noise in each dimension
      % Returns:
      %   nloglik (scalar)
      %     value of the negative log marginal likelihood
      %   eg (struct)
      %     derivatives of hyperparameters in Euclidean spaces containing
      %     kernParam (n_param x 1 vector) hyperparameters for kernels
      %     icmParam (this.dimY x this.dimY matrix) hyperparameters for coregionalization matrix
      %     noiseParam (this.dimY x 1 vector) hyperparameters for noise

      numx = size(x, 2);
      n = numx*this.dimY;

      % compute kernel matrix K and derivatives w.r.t. hyperparameters
      if nargout > 1 
        [K,d_K_d_hyp,d_K_d_icm] = this.cov(x, x); 
      else
        K = this.cov(x, x); 
      end
      K = 0.5*(K + K');
      noisem = diag(reshape(repmat(sigma.^2,1,numx)',1,[]));
      Ky = K + noisem;
   
      L = chol(Ky);
      alpha = L\(L'\y');        
      
      varargout{1} = 0.5*y*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi);
      
      if nargout > 1    
        inv_KyT = inv(Ky);
        num_kernp = size(d_K_d_hyp,2)/n;
        alphaT = alpha';
        dimY2 = this.dimY*this.dimY;

        % derivative of negative log likelihood w.r.t. 
        dtr_kernp = 0.5*sum(reshape(sum(repmat(inv_KyT,1,num_kernp).*d_K_d_hyp,1),n,num_kernp),1)';
        d_nloglik_d_kernp = -0.5*sum(reshape(alphaT*d_K_d_hyp,n,num_kernp).*alpha,1)' + dtr_kernp;

        dtr_icm = 0.5*reshape(sum(reshape(sum(repmat(inv_KyT,1,dimY2).*d_K_d_icm,1),n,dimY2),1),this.dimY,this.dimY);
        d_nloglik_d_icm = -0.5*reshape(sum(reshape(alphaT*d_K_d_icm,n,dimY2).*alpha,1),this.dimY,this.dimY) + dtr_icm;

        d_Ky_d_sigma = zeros(n,n*this.dimY);
        for i = 1:this.dimY
          dsigma2 = zeros(n,1);
          dsigma2((i-1)*numx+1:i*numx) = 2*sigma(i);
          d_Ky_d_sigma(:,(i-1)*n+1:i*n) = diag(dsigma2);
        end
        dtr_sigma = 0.5*sum(reshape(sum(repmat(inv_KyT,1,this.dimY).*d_Ky_d_sigma,1),n,this.dimY),1)';
        d_nloglik_d_sigma = -0.5*sum(reshape(alphaT*d_Ky_d_sigma,n,this.dimY).*alpha,1)' + dtr_sigma;

        eg.kernParam = d_nloglik_d_kernp;
        eg.icmParam = d_nloglik_d_icm;
        eg.noiseParam = d_nloglik_d_sigma;
        varargout{2} = eg;     
      end
    end    
  end
end