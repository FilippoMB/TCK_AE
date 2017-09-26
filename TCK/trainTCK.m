function [ res, C, G ] = trainTCK( X, varargin )
% trainTCK - Train the TCK
%
% INPUTS
% X: data array of size N x T x V, where N is the number of multivariate time series, T the length and V the number of attributes.
% minN: min percentage of subsample (optional)
% minV: min number of attributes for each GMM (optional)
% maxV: max number of attributes for each GMM  (optional)
% minT: min length of time segments for each GMM (optional)
% maxT: max length of time segments for each GMM (optional)
% C: max number of mixture components for each GMM (optional)
% G: number of randomizations for each number of components (optional)
% I: number of iterations (optional)
%
% OUTPUTS
% res: A cell of size ((C-1)*G,6) that for each q = 1:(C-1)*G contain 
	% Q: cluster posterior probabilities
	% mu: cluster means (time dependant + variable dependant)
	% s2: cluster variances (variable dependant)
	% theta: cluster priors
	% dim_idx: indexes of the subset of dimension considered 
	% time_idx: indexes of the subset of time intervals considered
% C
% G

%
% Reference: "Time Series Cluster Kernel for Learning Similarities between Multivariate Time Series with Missing Data", 2017 Pattern Recognition, Elsevier.
% Authors: "Karl �yvind Mikalsen, Filippo Maria Bianchi"

N = size(X,1); % number of time series
T = size(X,2); % time steps in each time series
V = size(X,3); % number of variables in each time series

% Parse the optional parameters
p = inputParser();
if(N < 100)
    p.addParameter('C', 10, @(z) assert(z>=2, 'C must be larger than 1'));
else
    p.addParameter('C', 40, @(z) assert(z>=2, 'C must be larger than 1'));
end
p.addParameter('G', 30);
p.addParameter('minN', 0.8, @(z) assert(z>0 && z<=1, 'The minimum percentage of subsample must be in (0,1]'));
if(V==1)
    p.addParameter('minV', 1, @(z) assert(z>=1 && z<=V, 'The minimum number of variables must be in [1,V]'));
else
    p.addParameter('minV', 2, @(z) assert(z>=1 && z<=V, 'The minimum number of variables must be in [1,V]'));
end
p.addParameter('maxV', min(ceil(0.9*V),15), @(z) assert(z>=1 && z<=V, 'The maximum number of variables must be in [1,V]'));
p.addParameter('minT', 6, @(z) assert(z>=1 && z<=T, 'The minimum length of time segments should be in [1,T]'));
p.addParameter('maxT', min(floor(0.8*T),25), @(z) assert(z>=1 && z<=T, 'The maximum length of time segments should be in [1,T]'));
p.addParameter('I', 20);
p.parse(varargin{:});
C = p.Results.C;
G = p.Results.G;
minN = p.Results.minN;
minV = p.Results.minV;
maxV = p.Results.maxV;
minT = p.Results.minT;
maxT = p.Results.maxT;
I = p.Results.I;


res = cell(G*(C-1),6);

% Check if there is missing data in the dataset.
nan_idx = isnan(X);
if(sum(sum(sum(nan_idx)))>0)
    missing = 1;
    fprintf('The dataset contains missing data\n\n');
else
    missing = 0;
    fprintf('The dataset does not contain missing data\n\n');
end

fprintf(' Training the TCK using the following parameters:\n C = %d, G =%d\n Number of MTS for each GMM: %d - %d (%d - 100 percent)\n Number of attributes sampled from [%d, %d]\n Length of time segments sampled from [%d, %d]\n\n', C, G, floor(minN*N), N, floor(minN*100), minV, maxV, minT, maxT);  

parfor i=1:G*(C-1)
    c= floor((i-1)/G) + 2;
    [o1, o2 , o3, o4, o5, o6] = GMM_MAP_EM(X,'C',c,'minN',minN,'minT',minT,'maxT',maxT,'minV',minV,'maxV',maxV,'I',I,'missing',missing);
    [res(i,:)] = {o1, o2 , o3, o4, o5, o6};
end



end

