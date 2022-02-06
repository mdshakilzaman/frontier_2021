function [tree,mergeUnknownId,mergeUnknownVal] = retractClusterSplit(tree,mergeIdParent,mergeUnknownId,mergeUnknownVal)
% merging leaf nodes into theri parent node
load hierarchy
if(~isempty(mergeIdParent>0))
    for i = 1:length(mergeIdParent)
        idPar  = mergeIdParent(i);
        cidChi = tree.children(:,idPar);
        idChi(i,:)= [find(tree.id==cidChi(1)) find(tree.id==cidChi(2))];
        
        % delete children row in parent
        tree.children(:,idPar)=-1;
        idxdup=find(mergeUnknownId==tree.id(idPar));
        mergeUnknownId(idxdup)=[];
        mergeUnknownVal(idxdup)=[];
        mergeUnknownId=[mergeUnknownId tree.id(idPar)];
        mergeUnknownVal=[mergeUnknownVal tree.val(idPar)];
    end
    
    idChi=idChi(:);
    % delete childrens
    tree.id(idChi)=[];
    tree.val(idChi)=[];
    tree.parent(idChi)=[];
    tree.children(:,idChi)=[];
end
clear hierarchy;
end