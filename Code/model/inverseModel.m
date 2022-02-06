function [t,invState] = inverseModel(simu,par)
% descriptopn: model evaluation
t   = [];
sta = [];
opt=odeset('NormControl','on');
% the spatially and temporally varying EP model 
[t,state] = ode45(@f_p,[0,175],simu.X0,opt,simu.s,simu.dim,par);
state = state(:,1:simu.dim)';
% the inverse model
invState=simu.H*(state);
end

