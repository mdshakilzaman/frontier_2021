function [data_lhx,data_lhf]=LHSample(numLHSamp,pdim,nclus,map,l,u,obs,simu)
% generate random points in the search space, evaluate the model
tic
% Start matlab's parallel pool, If no pool, do not create new one.
parPool = gcp('nocreate'); 
if ~isempty(parPool)
delete(gcp);
end
% pick random points in search space
data_lhx=l+(u-l)*lhsdesign(numLHSamp,pdim); 
% model evaluation in parallel
spmd 
    temp_lhf=zeros(numLHSamp,1);
    for ij =  labindex:numlabs:numLHSamp
        temp_lhf(ij) = -logPosterior(data_lhx(ij,:)',obs,simu,nclus,map);
    end   
end
% combine the model evaluation into single matrix
data_lhf=zeros(numLHSamp,1);
for i = 1 : size(temp_lhf,2)
    data_lhf = [temp_lhf{i}]+data_lhf;
end
delete(gcp)
toc
end