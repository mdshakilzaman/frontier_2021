function p = initPara(corMfree,ahaMap,lrvMap,hierarchy)
% generate a true parameter setting in given heart
% Input:
% corMfree : 3D coordinates of heart mesh
% ahaMap: American Heart Association (AHA) Division of heart
% lvrMap: determine left and right side of heart
% hierarchy : full coarse to fine hierarchy
% Output: parameter for each node as a vector p

dim = size(corMfree,1);
ldim = length(find(lrvMap==1));
rdim = length(find(lrvMap==0));
p = zeros(dim,1);
hParam =0.15; % healthy parameter value
lb = 0;
ub = 0.001;

% select an option for initialization of diseased parameter region
opt = input('select 1 or 2: 1.AHA heart 2.Pick Coordinate 3. From cluster ');
switch opt
    case 1
        [p,uhMap] = initAHA(); % initialize a segment in AHA as abnormal
    case 2
        [p,uhMap] = initCor(); % pick a coordinate, then a radious
    case 3
        [p,uhMap] = initClus(); % initialize in the original hierarchy
    otherwise
        disp('You must select 1 or 2 or 3!')
end

%%%% plot the parameter setting
udMap=unique(round(p,2));
ndMap=length(udMap);
cdmap = hsv(ndMap);

figure;hold on;
for id = 1:ndMap
    Indx = round(p,2)==udMap(id);
    plot3(corMfree(Indx,1),corMfree(Indx,2),corMfree(Indx,3),'.','MarkerSize',30,...
        'DisplayName',sprintf('Cluster %i',udMap(id)));
end
legend('show');

% total coverage of abnormal region in heart
covHeart = length(find(uhMap == 1))*100/dim;
% total coverage of abnormal region in in Right Ventricle
covLvRv = 2*uhMap-lrvMap;
covRV = length(find(covLvRv == 2))*100/rdim;
% total coverage of abnormal region in left ventricle
covLV = length(find(covLvRv == 1))*100/ldim;

disp('---------------------------------------------');
disp(['% of heart that is abnormal : ', num2str(covHeart)]);
disp(['% of LV that is abnormal : ', num2str(covLV)]);
disp(['% of RV that is abnormal : ', num2str(covRV)]);
str = input('Do you want to save the file y/n?','s')
if(lower(str)=='y')
    parTrue=p;
    save('parTrue.mat','parTrue');
    save('parCoverage.mat','covHeart','covLV','covRV');
    disp('------complete writing -----------------');
end
close all;

    function [p,uhMap] = initAHA()
        uhSeg= input('Enter segments numbers that are diseased, segment number range from 1 to 18,  [1 X n]:');
        uhParam = input('Enter parameter value for diseased segments [1 X n]:');
        uhSeg = uhSeg-1;
        s = 0;
        
        for i = 1 : dim
            for j = 1 : length(uhSeg);
                if (ahaMap(i)==uhSeg(j))
                    uhMap(i)=1;
                    p(i) =  uhParam(j)+ lb+ub*rand(1,1)*(-1)^(randi([0,1],1));
                    s=1;
                end
            end
            if(s==0)
                uhMap(i)=0;
                p(i) =  hParam+  lb+ub*rand(1,1)*(-1)^(randi([0,1],1));
            end
            s=0;
        end
        p=p';
        uhMap=uhMap';
    end

    function [p,uhMap]=initCor()
        fig = figure;
        hold on;
        scatter3(corMfree(:,1),corMfree(:,2),corMfree(:,3),5)
        dcm_obj=datacursormode(fig);
        %select center
        disp('Select a coordinate for center of abnormal region and press enter !!')
        pause
        center = getCursorInfo(dcm_obj);
        center =center.Position;
        scatter3(center(1),center(2),center(3),8,'r','filled')
        %select radious
        dcm_obj=datacursormode(fig);
        disp('Select a coordinate for radious of abnormal region and press enter !!')
        pause
        rad = getCursorInfo(dcm_obj);
        rad=rad.Position;
        rad = sqrt(sum((center - rad) .^ 2));
        % enter value of abnormal parameter
        uhParam =  input('Enter parameter value for diseased segments (0.4-0.5):');
        pos=sqrt(sum((bsxfun(@minus,corMfree,center).^2),2));
        uind=find(pos<=rad);
        p =  hParam+ lb+ub*rand(dim,1).*(-1).^(randi([0,1],dim,1));
        p(uind)=uhParam+ lb+ub*rand(length(uind),1).*(-1).^(randi([0,1],length(uind),1));
        uhMap = zeros(dim,1);
        uhMap(uind)=1;
        
    end


    function [p,uhMap]=initClus()
        %%% display the 17 segment map
        uMap=unique(ahaMap);
        nMap=length(uMap);
        cmap = hsv(nMap);
        
        %%%% plot with legend
        fig=figure;hold on;
        for i = 1:nMap
            Idx = ahaMap==uMap(i);
            plot3(corMfree(Idx,1),corMfree(Idx,2),corMfree(Idx,3),'.','MarkerSize',30,...
                'DisplayName',sprintf('Cluster %i',uMap(i)+1));
        end
        
        legend('show');
        
        %%% pick a point
        
        disp('Select a coordinate for abnormal location and press enter !!')
        pause
        dcm_obj=datacursormode(fig);
        center = getCursorInfo(dcm_obj);
        center =center.Position;
        
        %%%% load hierarchy
        [~,Idx]=min(pdist2(corMfree,center));
        uhParam = input('Enter parameter value for diseased segments(1 X n): ');
        side = input('Coverage in which side of heart: Left "1" ,right "2", whole heart "3" : ');
        givP = input('Enter % coverage: ')
        
        %%%
        if(side==2)
            total = length(find(lrvMap==0));
        elseif(side==1)
            total = length(find(lrvMap==1));
        else
            total=dim;
        end
        
        % total coverage in left ventricle
        curCov= length(hierarchy(Idx).mfree)*100/total;
        while(curCov<givP)
            Idx = hierarchy(Idx).parent;
            curCov= length(hierarchy(Idx).mfree)*100/total;
        end
        uind= hierarchy(Idx).mfree;
        p =  hParam+ lb+ub*rand(dim,1).*(-1).^(randi([0,1],dim,1));
        p(uind)=uhParam+ lb+ub*rand(length(uind),1).*(-1).^(randi([0,1],length(uind),1));
        uhMap = zeros(dim,1);
        uhMap(uind)=1;
        
    end
end