function [varargout] = hypertoroidalvMKernel(varargin)
  % Compute values and derivatives of the hypertoroidal von Mises kernel
  % Parameters:
  %   param ((D + (D+1)*D/2) x 1 vector)
  %     hyperparameters for defining the hypertoroidal von Mises kernel
  %     including kappa (D x 1 vector) and symmetric matrix A (D x D)
  %   x (D x nx vector)
  %     each column represents one of the inputs
  %   z (D x nz vector)
  %     each column represents one of the inputs
  % Returns:
  %   Kbase (nx x nz matrix)
  %     values of the kernel function evaluated at each pair of input (x,z)
  %   dhyp (nx x (D + (D+1)*D/2)*nz matrix)
  %     derivatives of the kernel function w.r.t. each hyperparameter
  
  if nargin < 2
    % compute number of hyperparameters for the kernel
    D = varargin{1}/2;
    n_Aparam = D;
    varargout{1} = D + n_Aparam; 
    return 
  end

  param = varargin{1};
  x_unit = varargin{2};  
  z_unit = varargin{3};

  [D,nx] = size(x_unit);
  D = D/2;
  n_Aparam = D;  
  nz = size(z_unit,2);
  n_xz = nx*nz;

  cosxz = zeros(n_xz,D);

  XT = x_unit';
  for i = 1:D
    idx = i:D:2*D;
    cosxz(:,i) = reshape(XT(:,idx)*z_unit(idx,:),n_xz,1);
  end

  % get hyperparameters
  kappa = param(1:D);
  w = param(D+1:D+n_Aparam);
  A = diag(w(1:D-1).*ones(D-1,1),-1);
  A(D,1) = w(D);
  A = A + A';

  % compute kernel matrix
  kappa_cos = reshape(cosxz*kappa,nx,nz);
  corel = reshape(sum(cosxz*A.*cosxz,2),nx,nz);

  Kbase = exp(kappa_cos + corel);
  varargout{1} = Kbase;
  
  if nargout > 1 
    Kcos = Kbase(:).*cosxz;
    % compute derivatives w.r.t. parameters in matrix A
    Kcos2 = 2*Kcos.*circshift(cosxz,[0,-1]);
    dhyp = reshape([Kcos,Kcos2],nx,nz*(D+n_Aparam));
    varargout{2} = dhyp;
  end
end