function [varargout] = icm(varargin)
  % Intrinsic coregionalization model for computing the kernel matrix for
  % multioutput Gaussian process
  % Parameters:
  %   fh_type (cell array)
  %     cell of function handles of kernel functions
  %   param (struct)
  %     hyperparameters for different kernels containing
  %     kernParam (n x 1 vector) hyperparameters for kernels
  %     icmParam (d x d matrix) hyperparameters for coregionalization matrix
  %   x (D x nx matrix)
  %     each column represents one of the inputs
  %   z (D x nz matrix)
  %     each column represents one of the inputs
  %   in_dim (integer)
  %     dimension of inputs required in each kernel
  %   out_dim (integer)
  %     dimension of outputs
  % Returns:
  %   K (nx x nz matrix)
  %     kernel matrix
  %   d_K_d_kernp ((nx*d) x (nz*d*n_param) matrix)
  %     derivatives of kernel functions w.r.t. hyperparameters for kernels
  %   d_K_d_B ((nx*d) x (nz*d^3) matrix)
  %     derivatives of kernel functions w.r.t. hyperparameters for coregionalization matrix

  fh_type = varargin{1};
  param = varargin{2};
  x = varargin{3};  
  z = varargin{4};  
  in_dim = varargin{5};

  % compute number of hyperparameters required in kernels
  [~,num_kernel] = size(fh_type);
  n_param = zeros(1,num_kernel);
  for j = 1:size(fh_type,2)
    fh = fh_type{j};
    n_param(j) = fh(in_dim);
  end
  n_param_sum = sum(n_param) + 1;
  
  sigma = param.kernParam(1);
  B = param.icmParam;

  x_num = size(x,2);
  z_num = size(z,2);      
  k = zeros(x_num,z_num,num_kernel);

  % for each kernel, compute values and derivatives at each pair of inputs
  if nargout > 1 
    d_hyp = zeros(x_num,z_num*n_param_sum);
  end

  idx_param_be = 2;
  idx_dim_be = 1;  
  for j = 1:num_kernel
    fh = fh_type{j};
    idx_param_ed = idx_param_be+n_param(j)-1;
    idx_dim_ed = idx_dim_be+in_dim-1;
    param_kern = [param.kernParam(idx_param_be:idx_param_ed)];
    if nargout == 1 
      k(:,:,j) = fh(param_kern,x(idx_dim_be:idx_dim_ed,:), ...
        z(idx_dim_be:idx_dim_ed,:));
    else
      col = 1+(idx_param_be-1)*z_num:idx_param_ed*z_num;
      [k(:,:,j),d_hyp(:,col)] = fh(param_kern,x(idx_dim_be:idx_dim_ed,:), ...
        z(idx_dim_be:idx_dim_ed,:));      
    end      
    idx_param_be = idx_param_be + n_param(j);
    idx_dim_be = idx_dim_be + in_dim;
  end

  % compute the product of multiple kernel matrices
  Kprod = prod(k,3);
  sigma2 = sigma^2;
  Kxx = Kprod*sigma2;
  varargout{1} = kron(B,Kxx);

  if nargout > 1 
    x_num2 = x_num^2;
    n_out = size(B,1);
    n_K = x_num*n_out;
    % derivatives of kernel matrix Kxx w.r.t. each hyperparameter
    d_k_d_kernp = sigma2*d_hyp;
    d_k_d_kernp(1:x_num,1:x_num) = 2*sigma*Kprod;

    idx_be = x_num2 + 1;
    for j = 1:num_kernel
      idx_ed = idx_be + x_num2*n_param(j) - 1;  
      idxk = ones(1,num_kernel,'logical');
      idxk(j) = 0;
      prodk = prod(k(:,:,idxk),3);
      d_k_d_kernp(idx_be:idx_ed) = reshape(d_k_d_kernp(idx_be:idx_ed),x_num,x_num*n_param(j)).*repmat(prodk,1,n_param(j));
      idx_be = idx_ed + 1;
    end          

    % derivatives of kernel matrix K = B \otimes Kxx w.r.t. each hyperparameter
    d_K_d_kernp = zeros(n_K,n_K*n_param_sum);
    for i = 1:n_out
      for j = 1:i
        row = 1+x_num*(i-1):x_num*i;
        a = reshape(repmat(n_K*(0:n_param_sum-1)',1,x_num)',1,x_num*n_param_sum);
        col = repmat(1+x_num*(j-1):x_num*j,1,n_param_sum) + a;            
        tmp = B(i,j)*d_k_d_kernp;
        d_K_d_kernp(row,col) = tmp;
        if j ~= i
          row = 1+x_num*(j-1):x_num*j;
          a = reshape(repmat(n_K*(0:n_param_sum-1)',1,x_num)',1,x_num*n_param_sum);
          col = repmat(1+x_num*(i-1):x_num*i,1,n_param_sum) + a;          
          d_K_d_kernp(row,col) = tmp;
        end
      end
    end

    % derivatives of kernel matrix K = B \otimes Kxx w.r.t. each paramater in B
    d_K_d_B = zeros(n_K,n_K,n_out,n_out);
    for i = 1:n_out
      for j = 1:i
        b = zeros(n_K);
        b((i-1)*x_num+1:i*x_num,(j-1)*x_num+1:j*x_num) = Kxx;
        d_K_d_B(:,:,i,j) = b;
        if j ~= i
          d_K_d_B(:,:,j,i) = b;
        end
      end
    end       

    varargout{2} = d_K_d_kernp;
    varargout{3} = reshape(d_K_d_B,n_K,[]);
  end
end    