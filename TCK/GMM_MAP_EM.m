function [ Q, mu, s2, theta, dim_idx, time_idx] = GMM_MAP_EM(X, varargin)
% MAP_EM - fit a GMM to time series data with missing values using MAP-EM
%
% INPUTS
% X: data array of size N x T x V
% C: number of mixture components (optional)
% minN: min percentage of subsample (optional)
% minV: min number of dimensions (optional)
% maxV: max number of dimensions (optional)
% minT: min length of time segments (optional)
% maxT: max length of time segments (optional)
% I: number of iterations (optional)
% missing: binary indicator. 1 if there is missing data and 0 if not
%
% OUTPUTS
% Q: cluster posterior probabilities
% mu: cluster means (time dependant + variable dependant)
% s2: cluster variances (variable dependant)
% theta: cluster priors
% dim_idx: indexes of the subset of dimension considered 
% time_idx: indexes of the subset of time intervals considered
%
% Reference: "Time Series Cluster Kernel for Learning Similarities between Multivariate Time Series with Missing Data", 2017 Pattern Recognition, Elsevier.
% Authors: "Karl �yvind Mikalsen, Filippo Maria Bianchi"

N = size(X,1); % number of time series
T = size(X,2); % time steps in each time series
V = size(X,3); % number of variables in each time series

% Parse the optional parameters
p = inputParser();
p.addParameter('minN', 0.8, @(z) assert(z>0 && z<=1, 'The minimum percentage of subsample must be in (0,1]'));
if(V==1)
    p.addParameter('minV', 1, @(z) assert(z>=1 && z<=V, 'The minimum number of variables must be in [1,V]'));
else
    p.addParameter('minV', 2, @(z) assert(z>=1 && z<=V, 'The minimum number of variables must be in [1,V]'));
end
p.addParameter('maxV', V, @(z) assert(z>=1 && z<=V, 'The maximum number of variables must be in [1,V]'));
p.addParameter('minT', 6, @(z) assert(z>=1 && z<=T, 'The minimum length of time segments should be in [1,T]'));
p.addParameter('maxT', min(floor(0.8*T),25), @(z) assert(z>=1 && z<=T, 'The maximum length of time segments should be in [1,T]'));
p.addParameter('C', 40);
p.addParameter('missing', 2);
p.addParameter('I', 20);
p.parse(varargin{:});
minN = p.Results.minN;
minV = p.Results.minV;
maxV = p.Results.maxV;
minT = p.Results.minT;
maxT = p.Results.maxT;
C = p.Results.C;
I = p.Results.I;
missing = p.Results.missing;

% Hyperparameters for mean prior (a0, b0) and the std dev prior (n0) of the mixture components
a0 = (1.0-0.001).*rand + 0.001;
b0 = (0.2-0.005).*rand + 0.005;
n0 = (0.2-0.001).*rand + 0.001;

% Randomly subsample dimensions, time intervals and samples
s = RandStream('mt19937ar','Seed',0);
if(N > 100)
    sN = randi([round(minN*N),N]);
else
    sN = round(0.9*N);
end
sub_idx = sort(randperm(s,N,sN)); % generate sN (sorted) integers between 1 and N

sV = randi([minV,maxV]);
dim_idx = sort(randperm(s,V,sV)); % generate sV (sorted) integers between 1 and V

t1 = randi([1,T-minT+1]);
t2 = randi([t1+minT-1,min(T,(t1+maxT-1))]);
sT = t2-t1+1;
time_idx = t1:t2; % generate sT contigous integers from t1 to t2
sX = X(sub_idx,time_idx,dim_idx);


if(missing == 1)
    nan_idx = isnan(sX);
    R = ones(size(sX));
    R(nan_idx)=0;

    % Calculate empirical moments
    mu_0 = zeros(sT,sV); % prior mean over time and variables (sT x sV)
    for v = 1:sV
        mu_0(:,v) = nanmean(sX(:,:,v),1);
    end
    s_0 = zeros(sV,1); % prior std over variables (sV x 1)
    tempX = reshape(sX,[sN*sT,sV]);
    for v = 1:sV
        s_0(v) = nanstd(tempX(:,v),0,1);
    end
    s2_0 = s_0.^2; 


    [S_0, invS_0] = deal(zeros(sT,sT,sV));
    T1 = repmat((1:sT)',[1,sT]);
    T2 = repmat((1:sT),[sT,1]);
    for v=1:sV
        S_0(:,:,v) = s_0(v)*b0*exp(-a0*(T1-T2).^2);
        if(rcond(S_0(:,:,v)) < 1e-8)  % check if the matrix can be inverted
            S_0(:,:,v) = S_0(:,:,v) + 0.1*S_0(1,1,v)*eye(sT);   %add a small number to the diagonal
        end
        invS_0(:,:,v) = inv(S_0(:,:,v));
    end


    % initialize model parameters
    theta = ones(1,C)/C;    % cluster priors        (1 x C)
    mu= zeros(sT,sV,C);     % cluster means         (sT x sV x C)
    s2 = zeros(sV,C);       % cluster variances     (sV x C)
    Q = zeros(sN,C);        % cluster assignments   (sN x C)

    sX(R==0) = -100000;

    for i=1:I

        % initialization: random clusters assignment
        if(i==1) 
            cluster = randi(C,[sN,1]);
            Q = double(bsxfun(@eq, cluster(:), 1:C));   

       % update clusters assignment
       else  
            for c=1:C
                distr_c = normpdf(sX, permute(repmat(mu(:,:,c),[1,1,sN]),[3,1,2]), permute(repmat(sqrt(s2(:,c)),[1,sN,sT]),[2,3,1]) ).^R;
                distr_c(distr_c < normpdf(3)) = normpdf(3);
                distr_c = reshape(distr_c,[sN,sV*sT]);
                Q(:,c) = theta(c)*prod(distr_c,2);
            end
            Q = Q./repmat(sum(Q,2),[1,C]);         
       end

        % update mu, s2 and theta
            for c=1:C
                theta(c) = sum(Q(:,c))/sN;
                for v=1:sV                          
                    var2 = sum(R(:,:,v),2)'*Q(:,c);
                    temp = (sX(:,:,v) - repmat(mu(:,v,c)',[sN,1]) ).^2;
                    var1 = Q(:,c)'*sum((R(:,:,v).*temp),2);
                    s2(v,c) = (n0*s2_0(v)+var1) / (n0+var2);

                    A =  invS_0(:,:,v) + diag(R(:,:,v)'*Q(:,c)/ s2(v,c));
                    b =  invS_0(:,:,v)*mu_0(:,v) + (R(:,:,v).*sX(:,:,v))'*Q(:,c)/s2(v,c);
                    mu(:,v,c) = A\b;
                end
            end        
    end % end for i=1:I

    % compute assignments for all data
    Q  = GMMposterior(X, C, mu, s2, theta, dim_idx, time_idx, missing );


%if no missing data the computations simplify a bit
elseif(missing == 0)
% Calculate empirical moments
    mu_0 = zeros(sT,sV); % prior mean over time and variables (sT x sV)
    for v = 1:sV
        mu_0(:,v) = mean(sX(:,:,v),1);
    end
    s_0 = zeros(sV,1); % prior std over variables (sV x 1)
    tempX = reshape(sX,[sN*sT,sV]);
    for v = 1:sV
        s_0(v) = std(tempX(:,v));
    end
    s2_0 = s_0.^2; 


    [S_0, invS_0] = deal(zeros(sT,sT,sV));
    T1 = repmat((1:sT)',[1,sT]);
    T2 = repmat((1:sT),[sT,1]);
    for v=1:sV
        S_0(:,:,v) = s_0(v)*b0*exp(-a0*(T1-T2).^2);
        if(rcond(S_0(:,:,v)) < 1e-8)  % check if the matrix can be inverted
            S_0(:,:,v) = S_0(:,:,v) + 0.1*S_0(1,1,v)*eye(sT);   %add a small number to the diagonal if S_0 is not invertible
        end
        invS_0(:,:,v) = inv(S_0(:,:,v));
    end



    % initialize model parameters
    theta = ones(1,C)/C;    % cluster priors        (1 x C)
    mu= zeros(sT,sV,C);     % cluster means         (sT x sV x C)
    s2 = zeros(sV,C);       % cluster variances     (sV x C)
    Q = zeros(sN,C);        % cluster assignments   (sN x C)

    for i=1:I

        % initialization: random clusters assignment
        if(i==1) 
            cluster = randi(C,[sN,1]);
            Q = double(bsxfun(@eq, cluster(:), 1:C));   

       % update clusters assignment
       else  
            for c=1:C
                distr_c = normpdf(sX, permute(repmat(mu(:,:,c),[1,1,sN]),[3,1,2]), permute(repmat(sqrt(s2(:,c)),[1,sN,sT]),[2,3,1]) );
                distr_c(distr_c < normpdf(3)) = normpdf(3);
                distr_c = reshape(distr_c,[sN,sV*sT]);
                Q(:,c) = theta(c)*prod(distr_c,2);
            end
            Q = Q./repmat(sum(Q,2),[1,C]);         
       end

        % update mu, s2 and theta
            for c=1:C
                sumQ = sum(Q(:,c));
                theta(c) = sumQ/sN;
                for v=1:sV
                    var2 = sT*sumQ;
                    var1 = Q(:,c)'*sum((sX(:,:,v) - repmat(mu(:,v,c)',[sN,1]) ).^2,2);
                    s2(v,c) = (n0*s2_0(v)+var1) / (n0+var2);

                    A =  invS_0(:,:,v) + (sumQ /s2(v,c))*eye(sT);
                    b =  invS_0(:,:,v)*mu_0(:,v) + (sX(:,:,v))'*Q(:,c)/s2(v,c);
                    mu(:,v,c) = A\b;
                end
            end        
    end % end for i=1:I

    % compute assignments for all data
    Q  = GMMposterior(X, C, mu, s2, theta, dim_idx, time_idx, missing );


else
    error('The value of the variable <missing> is not 0 or 1');
end



end

