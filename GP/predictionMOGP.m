function [m_est,covar] = predictionMOGP(num_kernel,in_dim,out_dim,xeopt,kernel_type_str,training_input,...
  obs,test_input)
  % Optimize and predict
  % Parameters:
  %   num_kernel (integer)
  %     number of kernel functions
  %   in_dim (integer)
  %     dimension of inputs required in each kernel
  %   out_dim (integer)
  %     dimension of outputs
  %   xe (1 x n_param vector)
  %     estimated hyperparameters from training
  %   kernel_type_str (string)
  %     type of kernels used for multioutput Gaussian processes
  %   training_points (in_dim x n matrix)
  %     each column represents one of the training inputs
  %   obs (1 x n*out_dim vector)
  %     observations at training points, a vector of all elements of an out_dim x n 
  %     matrix, taken row by row
  %   test_points (in_dim x m matrix)
  %     each column represents one of the test inputs
  %   gt (out_dim x m matrix)
  %     groundtruth at test points 
  % Returns:  
  %   err (scalar)  
  %     root mean squared error (RMSE) of estimates

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

  % compute predictive distributions at test points
  covfunc = @(x1,x2) icm(fh_cell,xeopt,x1,x2,in_dim,out_dim);
  gaussp = MultiOutputGaussianProcess(covfunc,out_dim);
  [m_est,covar] = gaussp.predict(training_input,obs,xeopt.noiseParam,test_input);
end    
