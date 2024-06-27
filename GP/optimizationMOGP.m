function [xe] = optimizationMOGP(num_kernel,in_dim,out_dim,x0,kernel_type_str,training_input,...
  obs)
  % Optimize and predict
  % Parameters:
  %   num_kernel (integer)
  %     number of kernel functions
  %   in_dim (integer)
  %     dimension of inputs required in each kernel
  %   out_dim (integer)
  %     dimension of outputs
  %   x0 (1 x n_param vector)
  %     initial hyperparameters 
  %   kernel_type_str (string)
  %     type of kernels used for multioutput Gaussian processes
  %   training_points (in_dim x n matrix)
  %     each column represents one of the training inputs
  %   obs (1 x n*out_dim vector)
  %     observations at training points, a vector of all elements of an out_dim x n 
  %     matrix, taken row by row
  % Returns:  
  %   xe (struct)
  %     hyperparameters

  % compute number of hyperparameters and get function handle for each kernel
  kernel_type = cell(1,num_kernel);
  kernel_type(:) = kernel_type_str;
  fh_cell = cell(1,num_kernel);
  num_hyp_k = 1;
  for j = 1:num_kernel
    fh = str2func(kernel_type{j});
    fh_cell{j} = fh;
    num_hyp_k = num_hyp_k + fh(in_dim);
  end

  % define manifolds of hyperparameters
  elements.kernParam = positivefactory(num_hyp_k,1);
  elements.icmParam = sympositivedefinitefactory(out_dim);
  elements.noiseParam = positivefactory(out_dim,1);
  problem.M = productmanifold(elements);
  
  % initialize hyperparameters
  x1.kernParam = x0(1:num_hyp_k)';
  x1.icmParam = reshape(x0(num_hyp_k+1:num_hyp_k+out_dim*out_dim),out_dim,out_dim);
  x1.noiseParam = x0(end-out_dim+1:end)';

  options.verbosity = 0;
  options.maxiter = 100;
  warning('off', 'manopt:getHessian:approx');

  % define cost function and gradient
  problem.cost = @(x) objfcn(x,training_input,obs,fh_cell,in_dim,out_dim);
  problem.egrad = @(x) egradMo(x,training_input,obs,fh_cell,in_dim,out_dim);

  % optimize
  tic;
  [xe] = trustregions(problem,x1,options);
  toc;
end    

function [f] = objfcn(x,training_input,obs,fh_cell,input_dim,output_dim)
  % Objective function for the optimization problem
  % Parameters:
  %   x (struct)
  %     hyperparameters
  %   training_points (in_dim x n matrix)
  %     each column represents one of the training inputs
  %   obs (1 x n*out_dim matrix)
  %     observations at training points
  %   fh_cell (cell)
  %     cell of function handles for kernel functions
  %   input_dim (integer)
  %     dimension of inputs required in each kernel
  %   output_dim (integer)
  %     dimension of outputs required in each kernel
  % Returns:
  %   f (scalar)
  %     cost of the objective function 

  cov = x.noiseParam;
  covfunc = @(x1,x2) icm(fh_cell,x,x1,x2,input_dim,output_dim);
  gaussp = MultiOutputGaussianProcess(covfunc,output_dim);
  f = gaussp.computeLogMarginalLik(training_input,obs,cov);
end   

function [eg] = egradMo(x,training_input,obs,fh_cell,input_dim,output_dim)
  % Gradients of the objective function w.r.t. hyperparameters in Euclidean spaces
  % Parameters:
  %   training_points (in_dim x n matrix)
  %     each column represents one of the training inputs
  %   obs (1 x n*out_dim matrix)
  %     observations at training points  
  %   fh_cell (cell array)
  %     cell of function handles
  %   in_dim (integer)
  %     dimension of inputs required in each kernel
  %   out_dim (integer)
  %     dimension of outputs  
  % Returns:
  %   eg (struct)
  %     struct of gradients of the objective function in Euclidean spaces 
  %     w.r.t. hyperparameters 

  cov = x.noiseParam;
  covfunc = @(x1,x2) icm(fh_cell,x,x1,x2,input_dim,output_dim);
  gaussp = MultiOutputGaussianProcess(covfunc,output_dim);
  [~,eg] = gaussp.computeLogMarginalLik(training_input,obs,cov);
end   