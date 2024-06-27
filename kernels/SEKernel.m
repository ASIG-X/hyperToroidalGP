function [varargout] = SEKernel(varargin)
  % Compute values and derivatives of the squared exponential kernel
  % Parameters:
  %   param (scalar)
  %     hyperparameter for defining the squared exponential kernel
  %   x (1 x nx vector)
  %     each column represents one of the inputs
  %   z (1 x nz vector)
  %     each column represents one of the inputs
  % Returns:
  %   Kbase (nx x nz matrix)
  %     values of the kernel function evaluated at each pair of input (x,z)
  %   dhyp (nx x nz matrix)
  %     derivatives of the kernel function w.r.t. each hyperparameter

  % compute number of hyperparameters for the kernel
  if nargin < 2, varargout{1} = 1; return 
  end

  % get hyperparameters
  l = varargin{1};

  x = varargin{2}; 
  z = varargin{3};  

  % compute kernel matrix
  xz2 = (x' - z).^2;
  Kbase = exp(-xz2/(2*l^2));
  varargout{1} = Kbase;

  if nargout > 1 
    varargout{2} = Kbase.*xz2/l^3;
  end
end