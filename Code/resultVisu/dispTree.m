function dispTree(tree,parTrue)
load hierarchy
n = length(tree.id);
treeSpec = zeros(1,n);
for idx = 1:n
    id = tree.id(idx);
    par= tree.parent(idx);
    labelId(idx)= id;
    labelVal(idx)=tree.val(idx);
    
    mfree= hierarchy(id).mfree;
    parMean = parTrue(mfree);
    parMean  = mean(parMean);
    labelTruePar(idx) = parMean;
    labelidx(idx)=idx;
    if(par==-1)
        idxp=0;
    else
        idxp = find(tree.id==par);
    end
    treeSpec(idx) = idxp;
end
[x, y, h, s]=trimtreelayout(treeSpec);
subplot(2,2,1)
trimtreeplot(treeSpec)
text(x, y, strread(num2str(round(labelTruePar,3)),'%s'), 'VerticalAlignment','bottom','HorizontalAlignment','right')
title({'Tree Splits wrt True value'},'FontSize',10,'FontName','Times New Roman');
subplot(2,2,2)
trimtreeplot(treeSpec)
text(x, y, strread(num2str(round(labelVal,3)),'%s'), 'VerticalAlignment','top','HorizontalAlignment','right')
title({'Tree Splits wrt Calc value'},'FontSize',10,'FontName','Times New Roman');

subplot(2,2,3)
trimtreeplot(treeSpec)
text(x, y, strread(num2str(labelId),'%s'), 'VerticalAlignment','bottom','HorizontalAlignment','right')
title({'Tree Splits wrt True value'},'FontSize',10,'FontName','Times New Roman');
subplot(2,2,4)
trimtreeplot(treeSpec)
text(x, y, strread(num2str(labelidx),'%s'), 'VerticalAlignment','top','HorizontalAlignment','right')
title({'Tree Splits wrt Calc value'},'FontSize',10,'FontName','Times New Roman');


end