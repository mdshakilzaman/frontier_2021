function [map, parUnknownId,parUnknownVal] = generateMap2(tree,dim,boundL,boundU)
% description : generate a mapping from low dimension to full cardiac mesh
% and coarse to fine multiple point initialization
% Output :
% map : mapping from the leaf nodes to MF nodes i.e. which MF nodes
% belong to which cluster
% parUnknownId: set of unknown to be optimized
% parUnknownValue: set of initial values for the unknown to be optimized

load hierarchy
map = zeros(dim,1);
c = find((tree.children(1,:)==-1));
parUnknownId=[];parUnknownVal=[];parUnknownVal1=[];
for i = c
    id = tree.id(i);
    val= tree.val(i);
    mfree = hierarchy(id).mfree;
    map(mfree) = id;
    parUnknownId = [parUnknownId id];
    if (val=='u') % if value of node undetermined
        idParent = tree.parent(i);
        valParent = tree.val(find(tree.id==idParent));
        parUnknownVal1= [parUnknownVal1 valParent];
        parUnknownVal=[parUnknownVal idParent];
    else
        parUnknownVal= [parUnknownVal val];
        parUnknownVal1= [parUnknownVal1 val];
    end
end
clear hierarchy;

%%% generate a set of initial point using convolution
pids=unique(parUnknownVal(find(parUnknownVal>1)));
num = length(pids);
for i = 1:num
    pval(i)=tree.val(find(tree.id==pids(i)));
end
ll = min(parUnknownVal1);
if(ll>0.14)
    ll=0.13;
end
a  = [0.12:0.03:0.5]';
a  = [a ; 100;200];
a  = repmat(a,1,num);
a  = num2cell([a],1);
vgrid = cell(1,numel(a));
[vgrid{:}] = ndgrid(a{:});
for i = 1:num % averaging(convolution)
    temp=[vgrid{i}];
    valn(:,2*i-1)=temp(:);
    valn(:,2*i)=(2*pval(i)-temp(:));
    idxx= find(valn(:,2*i-1)==100);
    valn(idxx,2*i-1)=pval(i);
    valn(idxx,2*i)=pval(i);
    idxx= find(valn(:,2*i-1)==200);
    valn(idxx,2*i-1)=0.15;
    valn(idxx,2*i)=0.15;
end
valn(any(valn<ll | valn>boundU,2),:)=[];
parUnknownValtemp=repmat(parUnknownVal,size(valn,1),1);
for i = 1:num
    ind=find(parUnknownVal==pids(i));
    parUnknownValtemp(:,ind(1))=valn(:,2*i-1);
    parUnknownValtemp(:,ind(2))=valn(:,2*i);
    
end
clear parUnknownVal;
parUnknownVal=parUnknownValtemp;
end
