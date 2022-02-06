function [fbest,xbest]= optimization(parUnknownVal,parUnknownId,map,obs,simu,dim,maxSamp,numLHSamp,boundL,boundU,path_out,tttt)
% optimization of 1d
global hyp;
global meanfunc;
global covfunc;
global likfunc;
global data_x;
global post;
global D;
global si;
pdim= length(parUnknownId);
covfunction={@covMaterniso, 5};
% fhTestfun = @(x) ucb(x); % acquisation function
 vX_l =  boundL*ones(pdim,1);  % lower bounds
 vX_u =  boundU*ones(pdim,1);  % upper bounds
% svOptions = struct('display', 'none', 'rho_beg',(boundU-boundL)/2-0.01,'rho_end', 1e-6, 'maxFunEval', 20000);

opt.algorithm = NLOPT_GN_DIRECT;
opt.lower_bounds = vX_l';
opt.upper_bounds = vX_u';
opt.min_objective = @ucb;
opt.ftol_abs = [1e-6];
opt.maxeval =[500];

% NLOPT_GN_DIRECT
% bounds = repmat([boundL boundU],pdim,1);
% opts   = [0.0001 500 250 1000 0];
% Problem.f = 'ucb';

cl =0.1;sigf=1;
if(pdim<=2)
  si = 0.16; cl=0.08;
elseif(pdim<=4)
    si =0.14; 
elseif(pdim<=8)
    si=0.12; 
else
    si=0.01;
end
parIniSamp=size(parUnknownVal,1);

%% initialize gaussian process

meanfunc = {@meanZero}; hyp.mean = [];
covfunc  = covfunction;  hyp.cov = [log(cl); log(sigf)];
likfunc  = {@likGauss}; sn = 0.001; hyp.lik = log(sn);
inffunc  = @infExact2;
data_x   = zeros(maxSamp+numLHSamp+parIniSamp,pdim);
data_f   =(zeros(maxSamp+numLHSamp+parIniSamp,1));
K        = sparse(zeros(maxSamp+numLHSamp+parIniSamp,maxSamp+numLHSamp+parIniSamp));
%% initialize
% LHSAmping
[data_x(1:numLHSamp,:), data_f(1:numLHSamp,:)]=LHSample(numLHSamp,pdim,parUnknownId,map,boundL,boundU,obs,simu);

% initialization
if(parIniSamp==1)
    data_x(numLHSamp+1,:)=parUnknownVal;
    data_f(numLHSamp+1,:) = -logPosterior(data_x(numLHSamp+1,:),obs,simu,parUnknownId,map);
else
    [data_x(1+numLHSamp:numLHSamp+parIniSamp,:), data_f(1+numLHSamp:numLHSamp+parIniSamp,:)]=initOptimization(parUnknownVal,pdim,parUnknownId,map,boundL,boundU,obs,simu);
end
D = numLHSamp+parIniSamp;



K(1:D,1:D)   = feval(@covMaterniso, 5, hyp.cov, full(data_x(1:D,:)));
[fbest,ibest]= max(data_f(1:D));
xbest = data_x(ibest,:);
post = feval(inffunc, hyp, meanfunc, covfunc, likfunc, data_x(1:D,:), data_f(1:D,:),K(1:D,1:D));
varlist={'varlist','cl','sigf','data_lhx'};
clear(varlist{:});



%% optimization
%newpoint = 1;
i = 0;
r=0;R=0;
%inddisp = 1 : pdim; %for display only
while(i<maxSamp)
    %  tic
    i=i+1  ;
%     [~, xmax] = bobyqa(fhTestfun, xbest', vX_l, vX_u, svOptions) ;
        [xmax, ~, ~] = nlopt_optimize(opt, xbest);
    % [~,xmax] = Direct(Problem,bounds,opts);
    
    
   % xmax=xmax';
    %[inddisp' xbest' xmax']
    % update K
    data_x(D+1,:) = xmax;
    fmax = -logPosterior(xmax,obs,simu,parUnknownId,map);
    data_f(D+1,:)=fmax;
    D = D+1;
    Ks = feval(@covMaterniso, 5,hyp.cov, full(data_x(1:D-1,:)), xmax);
    Cs = feval(@covMaterniso, 5,hyp.cov, xmax, 'diag');
    K(1:D-1,D) = Ks;
    K(D,1:D)=[Ks' Cs];
    post = feval(inffunc, hyp, meanfunc, covfunc, likfunc, full(data_x(1:D,:)), full(data_f(1:D,:)),full(K(1:D,1:D)));
    if(fmax>fbest)
        fbest = fmax;
        xbest = xmax;
        %ibest = D;
        %       numMax=0;
    end
    %numMax=numMax+1;
    
    if (D>45)
        %         if(numMax>150)
        %             break;
        %         end
        vs = sqrt(var(data_x(D-20:D,:)));
        cri=sum(vs<0.01);
        if(cri==pdim)
            break;
        end
    end
        r(i) = (2.1766e-09 - fbest);
    R(i) = sum(r)/i;
end
varlist= {'hyp','meanfunc','covfunc','likfunc','data_x','post','D','varlist'};
clear varlist;

figure(4)
hold on
plot(R,'-o');
plot(r,'-o');
title('Cumulative regret');
ylabel('function value')
xlabel('number of iteration')
legend(['R ' 'r' ],'Location','southeast')
save([path_out int2str(tttt) '_opt_data'])
hold off;


%save([path_out int2str(tttt) '_opt_data'])
%xbest=xbest';
end




