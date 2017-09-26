function [ Q ] = GMMposterior( X, C, mu, s2, theta, dim_idx, time_idx, missing )
%GMMposterior - Evaluate the posterior for the data X of the GMM described
%by C, mu, s2 and theta
%
% INPUTS
% X: data array of size N x V x T 
% C: number of mixture components (optional)
% mu: cluster means over time and variables (V x T)
% s2: cluster stds over variables (sV x 1)
% theta: cluster priors
% dim_idx: subset of variables to be used in the clustering
% time_idx: subset of time intervals to be used in the clustering
% missing: binary indicator. 1 if there is missing data and 0 if not
%
% OUTPUTS
% Q: posterior
%
% Reference: "Time Series Cluster Kernel for Learning Similarities between Multivariate Time Series with Missing Data", 2017 Pattern Recognition, Elsevier.
% Authors: "Karl �yvind Mikalsen, Filippo Maria Bianchi"

N = size(X,1); % number of time series

% initialize variables
Q = zeros(N,C);
sX = X(:,time_idx,dim_idx);
sV = length(dim_idx);
sT = length(time_idx);


if(missing == 1)
    nan_idx = isnan(sX);
    R = ones(size(sX));
    R(nan_idx)=0;
    sX(R==0) = -100000;

    for c=1:C
        distr_c = normpdf(sX, permute(repmat(mu(:,:,c),[1,1,N]),[3,1,2]), permute(repmat(sqrt(s2(:,c)),[1,N,sT]),[2,3,1]) ).^R;
        distr_c(distr_c < normpdf(3)) = normpdf(3);
        distr_c = reshape(distr_c,[N,sV*sT]);
        Q(:,c) = theta(c)*prod(distr_c,2);
    end
    Q = Q./repmat(sum(Q,2),[1,C]);

elseif(missing == 0)
    for c=1:C
        distr_c = normpdf(sX, permute(repmat(mu(:,:,c),[1,1,N]),[3,1,2]), permute(repmat(sqrt(s2(:,c)),[1,N,sT]),[2,3,1]) );
        distr_c(distr_c < normpdf(3)) = normpdf(3);
        distr_c = reshape(distr_c,[N,sV*sT]);
        Q(:,c) = theta(c)*prod(distr_c,2);
    end
    Q = Q./repmat(sum(Q,2),[1,C]);
    
else
    error('The value of the variable missing is not 0 or 1');
end


end

