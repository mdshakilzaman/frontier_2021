function showResult(parOptVal,parUnknownId,map,corMfree,boundL,boundU,tttt,path_out,parTrue)
dim =size(corMfree,1);
parOptVal=mapParam(parOptVal);

%% QUANTITATIVE RESULT

% Consistency Metric
S2=find(parTrue>0.151);
S1=find(parOptVal>=0.151);
CoM=length(intersect(S1,S2))/length(union(S1,S2));
% Correlation Coefficient
precision =length(intersect(S1,S2))/length(S1)
recall =length(intersect(S1,S2))/length(S2)



% Accuracy
diff= abs(parTrue-parOptVal);
limit =[0.001 0.01 0.02 0.1];
for i = 1:length(limit)
    accuracy(i) = sum(diff<limit(i))*100/dim;
end
disp('** RESULT **')
disp(CoM)
disp([limit; accuracy])

%% VISUAL RESULT
% write the parameter to file for paraview
% filename=[path_out 'parameter' int2str(tttt) '.bin']
% fidw = fopen(filename,'wb');
% fwrite(fidw,parOptVal,'double');
% fclose(fidw);

plotError



    function plotError
        %  figure('Position',[300 700 300 500])
        
        colormap jet
        subplot(3,1,1)
        scatter3(corMfree(:,1),corMfree(:,2),corMfree(:,3),10,parOptVal,'filled')
        set(gca, 'CLim', [boundL, boundU]);
        colorbar
        axis equal;axis off;view(3);cameramenu
        
        subplot(3,1,2)
        scatter3(corMfree(:,1),corMfree(:,2),corMfree(:,3),10,parTrue,'filled')
        set(gca, 'CLim', [boundL, boundU]);
        colorbar
        axis equal;axis off;view(3);cameramenu
        
        % colormap gray
        subplot(3,1,3)
        scatter3(corMfree(:,1),corMfree(:,2),corMfree(:,3),10,(diff/(boundU-boundL))*100,'filled')
        set(gca, 'CLim', [0, 100]);
        colorbar
        axis equal;axis off;view(3);cameramenu
        %suptitle('truth, calcualted, error')
        
    end

    function px1=  mapParam(px0)
        %  px1= zeros(dim,1);
        px1=map;
        for ij = 1: length(parUnknownId)
            px1(find(map==parUnknownId(ij)))=px0(ij);
        end
        
    end



end