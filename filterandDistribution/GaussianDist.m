classdef GaussianDist
    properties
        mu (:,1) double % mean
        C (:,:) double % covariance, C=sigma^2 in 1D
        dim
    end
    
    methods
        function this = GaussianDist(mu_, C_, checkValidity)
            % Constructor
            %
            % Parameters:
            %   mu_ (d x 1)
            %       location parameter (mean vector)
            %   C_ (d x d)
            %       covariance matrix
            arguments
                mu_ (:,1) double {mustBeNonempty}
                C_ (:,:) double {mustBeNonempty}
                checkValidity (1,1) logical = true % Can disable because it is not a cheap operation
            end
            assert(size(mu_,1)==size(C_,1), 'size of C invalid')
            assert(size(mu_,1)==size(C_,2), 'size of C invalid')
            this.dim = size(mu_,1);
            this.mu = mu_;
            
            % check that C is positive definite
            dim = length(mu_);
            if checkValidity&&dim==1
                assert(C_>0, 'C must be positive definite');
            elseif checkValidity&&dim==2
                assert(C_(1,1)>0 && det(C_)>0, 'C must be positive definite');
            elseif checkValidity
                chol(C_); % will fail if C_ is not pos. def.
            end
            
            this.C = C_;
        end
        
        function p = pdf(this, xa)
            % Evaluate pdf at each column of xa
            %
            % Parameters:
            %   xa (d x n)
            %       n locations where to evaluate the pdf
            % Returns:
            %   p (1 x n)
            %       value of the pdf at each location
            arguments
                this (1,1) GaussianDist
                xa double {mustBeNonempty}
            end
            assert(size(xa,1)==size(this.mu,1), 'dimension incorrect')
            
            p = mvnpdf(xa',this.mu',this.C)'; % just for me...
        end
        
        function samples = sample(this, n)
            % Obtain n samples from the distribution
            %
            % Parameters:
            %   n (scalar)
            %       number of samples
            % Returns:
            %   s (d x n)
            %       one sample per column
            arguments
                this (1,1) GaussianDist
                n (1,1) {mustBeInteger,mustBePositive}
            end
            samples = mvnrnd(this.mu, this.C, n)';
        end
    end
end
