function [map, parUnknownId,parUnknownVal] = generateMap(tree,dim)
% description : generate a mapping from low dimension to full cardiac mesh 
% and coarse to fine single point initialization
% Output :
% map : mapping from the leaf nodes to MF nodes i.e. which MF nodes
% belong to which cluster
% parUnknownId: set of unknown to be optimized
% parUnknownValue: set of initial values for the unknown to be optimized


load hierarchy
map = zeros(dim,1); % map from leaves to cardaic mesh, maps which node belongs to which cluster(i.e leaf)
c = find((tree.children(1,:)==-1)); %the leaves
parUnknownId=[];parUnknownVal=[];
% for each leaf node, setup initial parameter value and map meshfree nodes
for i = c
    id = tree.id(i); % nodes id in tree
    val= tree.val(i); %  nodes value in tree
    mfree = hierarchy(id).mfree; % meshfree nodes corresponding to the node
    map(mfree) = id; % generate map
    parUnknownId = [parUnknownId id]; % set of leaf nodes
    if (val=='u') %  node has no previous value, means it was split/refined    
        idParent = tree.parent(i);
        valParent = tree.val(find(tree.id==idParent));
        parUnknownVal= [parUnknownVal valParent]; % values initialization from parent
    else
        parUnknownVal= [parUnknownVal val]; % old nodes have values
    end
end
end