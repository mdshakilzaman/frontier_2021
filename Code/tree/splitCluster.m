function [tree,split] = splitCluster(tree,mergeIdParent,splitIdParent,splitCidParent)
% splitting the leaf nodes
load hierarchy;
split=0;
for i = 1:length(splitIdParent)
    idPar  = splitIdParent(i);
    cidPar = splitCidParent(i);
    cidChi = hierarchy(cidPar).children;
    if (cidChi==-1)
        continue;
    end
    split=1;
    % update parent
    tree.children(:,idPar) = cidChi;
    % add children nodes
    idNew= length(tree.id)+1;
    tree.id(idNew)=cidChi(1);
    tree.parent(idNew) = cidPar;
    tree.children(:,idNew) =-1;
    tree.val(idNew)='u';
    tree.mfree(idNew)= length(hierarchy(tree.id(idNew)).mfree);
    
    idNew=idNew+1;
    tree.id(idNew)=cidChi(2);
    tree.parent(idNew) = cidPar;
    tree.children(:,idNew) =-1;
    tree.val(idNew)='u';
    tree.mfree(idNew)= length(hierarchy(tree.id(idNew)).mfree);
    
end
clear hierarchy;
end