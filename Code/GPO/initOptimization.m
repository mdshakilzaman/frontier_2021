function [data_lhx,data_lhf]=initOptimization(parUnknownVal,pdim,nclus,map,l,u,obs,simu)
% description: evaluate model at the coarser scale points
numLHSamp=size(parUnknownVal,1);
% being matlab's parallel pool
parPool = gcp('nocreate'); % If no pool, do not create new one.
if ~isempty(parPool)
delete(gcp);
end
% the coarser scale points
data_lhx=parUnknownVal;%l+(u-l)*lhsdesign(numLHSamp,pdim);
% evaluate model in parallel for the coarser scale points
spmd
    temp_lhf=zeros(numLHSamp,1);

    for ij =  labindex:numlabs:numLHSamp
        temp_lhf(ij) = -logPosterior(data_lhx(ij,:)',obs,simu,nclus,map);
    end
    
end
% combine the data into one matrix
data_lhf=zeros(numLHSamp,1);
for i = 1 : size(temp_lhf,2)
    data_lhf = [temp_lhf{i}]+data_lhf;
end
delete(gcp)
end