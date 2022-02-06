function  makeHierarchy()
%input corMfree is a n X 3, 3D coordinates of heart
load('modelFiles.mat','corMfree','ahaMap');

%%%% generate the hierarchy with hierarchial clustering
dim = size(corMfree,1);
link = 'average'; %  linkage metric
distn ='euclidean'; % distance metric
T = linkage(corMfree,link,distn);

%%%%%%% visualize the clustering
% c = cluster(T,'maxclust',5);
% uMap=unique(c);
% nMap=length(uMap);
% cmap = hsv(nMap);
% 
%%%%plot with legend
% figure,hold on
% for i = 1:nMap
%     Idx = c==uMap(i);
%     plot3(corMfree(Idx,1),corMfree(Idx,2),corMfree(Idx,3),'.','MarkerSize',20,...
%        'DisplayName',sprintf('Cluster %i',uMap(i)+1));
% end
% legend('show');
% title('clustering result')

%%% savung the hierarchy in matlab structure for later reference
for i = 1: dim
    hierarchy(i).id = i;
    hierarchy(i).parent = -1;
    hierarchy(i).children = -1;
    %hierarchy(i).cor = corMfree(i,:);
    hierarchy(i).mfree = i;
end
for i = 1 : dim-1
    cluster1  = T(i,:);
    child1 = cluster1(1);
    child2 = cluster1(2);
    id = i+dim;
    hierarchy(id).id = id;
    hierarchy(id).parent = -1;
    hierarchy(id).children = [child1; child2];
    %hierarchy(id).cor =  [hierarchy(child1).cor; hierarchy(child2).cor];
    hierarchy(id).mfree =  [hierarchy(child1).mfree; hierarchy(child2).mfree];
    hierarchy(child1).parent = [id];
    hierarchy(child2).parent = [id];
    varlist = {'child1','child2','id','cluster'};
    clear varlist;
end
save('hierarchy.mat','hierarchy');
end