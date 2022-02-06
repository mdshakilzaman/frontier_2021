function [mergeIdParent,splitCidChildren,splitIdChildren,tree  ] = findGainlessClusterSplit(tree,obs,simu,map,fOptVal,mergeUnknownId,mergeUnknownVal)
% calculate the gain of split and determine: 1) leaves to split 2) leaves to retract 
% Find all the leaf nodes, leaf nodes have no children
idChildren = find(tree.children(1,:)==-1); % the leaves id
cidParent = tree.parent(idChildren); % parents of leaf
numChildren = length(idChildren);
idParent=zeros(1,numChildren);
for i = 1:numChildren
    idParent(i)=find(tree.id==cidParent(i));
end
[idParent,ordInd]=sort(idParent);

%% Find out which leaf has sibling and which does not have sibling
% idparent2 has 2 children, idParent1 has 1 Child
% gain is calculated for idParent2 only, other Gain for idParent1
[a,b]=histc(idParent,unique(idParent));
counts=a(b);
idParent2=idParent;idParent2(counts==1)=[];
idParent1=idParent;idParent1(counts~=1)=[];

valParent=tree.val(idParent);
cidParent=tree.id(idParent);
%mfreeParentTemp = tree.mfree(idParent);

idChildren=idChildren(ordInd);
valChildren =tree.val(idChildren);
cidChildren=tree.id(idChildren);

%% Gain for cluster with no sibling
numParent1=length(unique(idParent1));%length(idParent)/2;
temp1 = valChildren';
temp1 = repmat(temp1,1,numParent1);
uCids=[];
for i = 1:numParent1
    indices=find(idParent==idParent1(i))';
    uCid = cidChildren(indices);
    temp1(indices,i)=mergeUnknownVal(find(mergeUnknownId==uCid));
    % mfreeParent(i) =mfreeParentTemp(2*i);
    uCids=[uCids uCid];
end

f1=zeros(numParent1,1);
for ij =  labindex:numlabs:numParent1
    f1(ij) = -logPosterior(temp1(:,ij)',obs,simu,cidChildren,map);
end
dfGain1= fOptVal-f1; % gain due to chage in its value
[mi] = uCids(find(dfGain1<10^-3)); % no gain by optimization
si=find(dfGain1>0); % gain>0 by optimization

% if no gain by change in value use the old value
for ij = 1: length(mi) 
    idx = find(cidChildren==(mi(ij)));
    valChildren(idx)=mergeUnknownVal(find(mergeUnknownId==mi(ij)));
end
if(length(mi)>0)
    fOptVal= -logPosterior(valChildren',obs,simu,cidChildren,map);
    tree=updateTreeValue(tree,cidChildren,valChildren);
end

%% Gain for cluster with two sibling
numParent2=length(unique(idParent2));%length(idParent)/2;
temp2 = valChildren';
temp2 = repmat(temp2,1,numParent2);
for i = 1:numParent2
    indices=find(idParent==idParent2(2*i))';
    temp2(indices,i)=valParent(indices(1));
    % mfreeParent(i) =mfreeParentTemp(2*i);
end
f2=zeros(numParent2,1);
for ij =  labindex:numlabs:numParent2
    f2(ij) = -logPosterior(temp2(:,ij)',obs,simu,cidChildren,map);
end

% the gain
dfGain2= fOptVal-f2; 
dfGainP=dfGain2*100/sum(dfGain2);
% merge the one with negligible gain

% split the one with maximum gain
[mGain,~]= max(dfGain2); 
mergeIndex=find(dfGainP<=5); 
splitIndex=find(dfGain2==mGain);
%% Determine the id/cid of nodes to be split and merged in tree/hierarchy 
% find the maximum gain between the leaf node with sibling and no sibling
mmm=-1;
for i = 1:length(si)
    if(dfGain1(si(i))>mGain)
        mmm=si(i);
        break;
    end
end
% calcuate the id of the nodes to split
if(mmm==-1 )
    splitCidChildren=[];
    splitIdChildren=[];
    splitIdParent = idParent2(2*splitIndex);
    for i = 1:length(splitIdParent)
    splitCidChildren=[splitCidChildren cidChildren(find(idParent==splitIdParent(i)))];
    splitIdChildren =[splitIdChildren idChildren(find(idParent==splitIdParent(i)))];
    end
    
else
    splitIdParent = idParent1(mmm);
    splitCidChildren=cidChildren(find(idParent==splitIdParent));
    splitIdChildren =idChildren(find(idParent==splitIdParent));
    % splitCidParent =
end

%%% Calcualte cid(id in the hierarchy) of the node to be merged
comIdx=find(mergeIndex==splitIndex);
mergeIndex(comIdx)=[];
mergeIdParent=idParent2(2*mergeIndex);

end