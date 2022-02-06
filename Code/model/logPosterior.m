function [lpos] = logPosterior(px0,obs,simu,parUnknownId,map)
% evaluate the obejctive function
par=mapParam(px0);
%% find the corresponding time instant in simulated bsp
[t,f] = inverseModel(simu,par);
tmstep=length(obs.time);
index = zeros(length(obs.time),1);%,'gpuArray');
for i  = 1: length(obs.time)
    [dummy index(i)] = min(abs(obs.time(i)-t));
end
f  = f(:,index);
f=f';obsbsp=obs.bsp';
% f=f(:,any(f)); obsbsp =obsbsp(:,any(obsbsp));
% corVal= mean(diag(corr(f,obsbsp))); % compute correlation coefficient
% compute sum of squared error
d = (f(:)-obsbsp(:));
loglik =  sum(d.*d);%/(120*length(obs.time));
lpos = loglik;%-corVal; % minimize this

    function px1=  mapParam(px0)
        % each MF node coresponds to a value in leaf cluster
        px1=map;
        for ij = 1: length(parUnknownId)
            px1(find(map==parUnknownId(ij)))=px0(ij);
        end
        
    end
end