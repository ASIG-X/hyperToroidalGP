function [varargout] = vMKernel(varargin)
  % Compute values and derivatives of the von Mises kernel
  % Parameters:
  %   param (scalar)
  %     hyperparameter for defining the von Mises kernel
  %   x (1 x nx vector)
  %     each column represents one of the inputs
  %   z (1 x nz vector)
  %     each column represents one of the inputs
  % Returns:
  %   Kbase (n_x x n_z matrix)
  %     values of the kernel function evaluated at each pair of input (x,z)
  %   dhyp (n_x x n_z matrix)
  %     derivatives of the kernel function w.r.t. each hyperparameter

  % compute number of hyperparameters for the kernel
  if nargin < 2, varargout{1} = 1; return 
  end

  % get hyperparameters
  kappa = varargin{1};

  x = varargin{2};  
  z = varargin{3};

  % compute kernel matrix
  XTX = cos(x' - z);
  Kbase = exp(kappa*XTX);   
  varargout{1} = Kbase;
  
  if nargout > 1 
    varargout{2} = Kbase.*XTX;
  end
end