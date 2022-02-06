function [fbest,xbest]= optimization(parUnknownVal,parUnknownId,map,obs,simu,dim,maxSamp,numLHSamp,boundL,boundU,path_out,tttt)
% description : Gaussian Process Bayesian Optimization(GP-BO)

global hyp;
global meanfunc;
global covfunc;
global likfunc;
global data_x;
global post;
global D;
global si;

%% initialize optimization parameters
fhTestfun = @(x) ucb(x);         % acquisation function
pdim= length(parUnknownId);      % unknown kernel
vX_l =  boundL*ones(pdim,1);     % lower bounds
vX_u =  boundU*ones(pdim,1);     % upper bounds
%svOptions = struct('display', 'none', 'rho_beg',(boundU-boundL)/2-0.01,'rho_end', 1e-6, 'maxFunEval', 20000);

opt.algorithm = NLOPT_LN_BOBYQA;
opt.lower_bounds = vX_l';
opt.upper_bounds = vX_u';
opt.min_objective = @ucb;
opt.ftol_abs = [1e-6];
opt.maxeval =  [20000];
%opt.initial_step =(boundU-boundL)/2-0.01;


%[xopt, fmin, retcode] = nlopt_optimize(opt, [1.234 5.678]);



%% initialize gaussian process
cl =0.1;sigf=0.002;  si =1;
% if(pdim<=2)
%     si = 0.16;% cl=0.08;
% elseif(pdim<=4)
%     si =0.14;
% elseif(pdim<=8)
%     si=0.12;
% else
%     si=0.01;
% end
parIniSamp=size(parUnknownVal,1);
covfunction={@covMaterniso, 5};  % kernel
meanfunc = {@meanZero}; hyp.mean = [];
covfunc  = covfunction;  hyp.cov = [log(cl); log(sigf)];
likfunc  = {@likGauss}; sn = 0.001; hyp.lik = log(sn);
inffunc  = @infExact2;
data_x   = zeros(maxSamp+numLHSamp+parIniSamp,pdim);
data_f   =(zeros(maxSamp+numLHSamp+parIniSamp,1));
K        = sparse(zeros(maxSamp+numLHSamp+parIniSamp,maxSamp+numLHSamp+parIniSamp));
%% initialize the GP
% First, initialize using few random points, Latin Hyper Cube Sampling
% using matlab's parallel tool box
[data_x(1:numLHSamp,:), data_f(1:numLHSamp,:)]=LHSample(numLHSamp,pdim,parUnknownId,map,boundL,boundU,obs,simu);

% initialization using the coarser scale points
if(parIniSamp==1) % only one coarser scale point
    data_x(numLHSamp+1,:)=parUnknownVal;
    data_f(numLHSamp+1,:) = -logPosterior(data_x(numLHSamp+1,:),obs,simu,parUnknownId,map);
else % multiple coarser scale points, use matlab's parallel toolbox for evaluation
    [data_x(1+numLHSamp:numLHSamp+parIniSamp,:), data_f(1+numLHSamp:numLHSamp+parIniSamp,:)]=initOptimization(parUnknownVal,pdim,parUnknownId,map,boundL,boundU,obs,simu);
end
D = numLHSamp+parIniSamp; % the number of samples so far
K(1:D,1:D)   = feval(@covMaterniso, 5, hyp.cov, full(data_x(1:D,:))); % evaluate the covariance matrix
[fbest,ibest]= max(data_f(1:D));
xbest = data_x(ibest,:); % the optimum so far
% computer the inverse of K and other GP properties
post = feval(inffunc, hyp, meanfunc, covfunc, likfunc, data_x(1:D,:), data_f(1:D,:),K(1:D,1:D));
varlist={'varlist','cl','sigf','data_lhx'};
clear(varlist{:});

%% GP-BO optimization start
i = 0;
r=0;
while(i<maxSamp)
    i=i+1;
    % maximize upper confidence bound (ucb) of GP
   % [~, xmax] = bobyqa(fhTestfun, xbest', vX_l, vX_u, svOptions) ;
    [xmax, ~, ~] = nlopt_optimize(opt, xbest);
    
    %xmax=xmax';
    % update GP using the new optimial point
    data_x(D+1,:) = xmax;
    fmax = -logPosterior(xmax,obs,simu,parUnknownId,map);
    data_f(D+1,:)=fmax;
    D = D+1;
    Ks = feval(@covMaterniso, 5,hyp.cov, full(data_x(1:D-1,:)), xmax);
    Cs = feval(@covMaterniso, 5,hyp.cov, xmax, 'diag');
    K(1:D-1,D) = Ks;
    K(D,1:D)=[Ks' Cs];
    post = feval(inffunc, hyp, meanfunc, covfunc, likfunc, full(data_x(1:D,:)), full(data_f(1:D,:)),full(K(1:D,1:D)));
    if(fmax>fbest) % the maximum so far
        fbest = fmax;
        xbest = xmax;
    end
    if (D>45) % convergence : the x values do not change much over 20 samples
        vs = sqrt(var(data_x(D-20:D,:)));
        cri=sum(vs<0.005);
        if(cri==pdim)
            break;
        end
    end
    r(i) = ( 7.0721e-04 - fbest);
    R(i) = sum(r)/i;
end
varlist= {'hyp','meanfunc','covfunc','likfunc','data_x','post','D','varlist'};
clear varlist;
% the cumulative regret

% figure(1)
% hold on
% plot(data_f(1:D),'-o');
% title('Change in function value wrt # of iterations & scale');
% ylabel('function value')
% xlabel('number of iteration')
% legend(['scale ' int2str(pdim) ],'Location','southeast')
% 
% 
% 
% save([path_out int2str(tttt) '_opt_data'])
% %xbest=xbest';

% figure(4)
% %hold on
% plot(R,'-o');
% hold on;
% plot(r,'-o');
% title('Cumulative regret');
% ylabel('function value')
% xlabel('number of iteration')
% legend('R ', 'r' ,'Location','southeast')
% save([path_out int2str(tttt) '_opt_data'])
% hold off;
% %xbest=xbest';
end




