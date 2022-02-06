function obs_gt= initBSP_new(truep)
% given a parameter setting simulate ECG data with noise
% trup: true parameter setting
addpath('/home/mz1482/project/BOVAE (miccai2018)/Code/', ...
'/home/mz1482/project/BOVAE (miccai2018)/Code/model', ...
'/home/mz1482/project/BOVAE (miccai2018)/Code/initialization');
load('modelFiles.mat','simu');
[timesteps,state] = inverseModel(simu,truep); % the simulation model
[data,sigPow,noiPow_SD2] = genIniNoi(state,30,'','',0); % add 30db Gaussian noise

% downsample the simulated data in time because the real ECG data is
% sparser than the simulated data
tsparse = 0:(170/450*0.5):170;
timestps = length(tsparse);
index = zeros(1,timestps);
for i  = 1: timestps
    [dummy,index(i)] = min(abs(tsparse(i)-timesteps));
end
obs_gt.bsp  = state(:,index);
obs_gt.time = timesteps(index);
obs_gt.noiseSigma = (repmat(noiPow_SD2,length(tsparse),1));
end