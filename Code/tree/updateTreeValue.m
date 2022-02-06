function tree= updateTreeValue(tree,parUnknownId,parOptVal)
% description: update the values of tree leaf nodes
% Input: 
%parUnknwonID: id of the leaf nodes,
%parOptVal: optimal value of leaves to update
n= length(parUnknownId);
for i = 1:n
    indChi = find(tree.id==parUnknownId(i)); % the leaves
    tree.val(indChi) =parOptVal(i); % set the value of leaf
end
end