clear all; close all;clc;
startup
%addpath('/Users/jd1336/.local/mex/');
% mex(strcat('-I"',pwd,'"'), 'bobyqa_alg.cpp')
%% read data
%path_in = ''; % path to input files: heart geometry, simualtion files, etc
path_out='../Result/'; % output path
%readFiles(path_in); % read all the necessary input files into matlabfile

%% generate Coarse to Fine Hierarchy
makeHierarchy();
%% load the necessary simulation input data
i=1;
load('modelFiles','corMfree','ahaMap','lrvMap');
load hierarchy;
%parTrue = initPara(corMfree, ahaMap,lrvMap,hierarchy); %% initialization
%of parameters
clear corMfree; clear ahaMap; clear lrvMap;
load parTrue
obs = initBSP(parTrue);  %%% simulate ECG with noise for the given parameter setting
%% BEGIN ALGORITHM
%%% initialize variables
load('modelFiles.mat','simu','corMfree');boundL=0; boundU=0.5;  dim = simu.dim; load hierarchy;
%%% initilize tree as root
root = max([hierarchy.id]);
tree.id(1) = hierarchy(root).id;
tree.parent(1) = hierarchy(root).parent;
tree.children(:,1) =[-1 -1]'; %%% no children
tree.val(1)=0.2; %%% intial parameter value of root node
tree.mfree(1) = length(hierarchy(root).mfree);%%% number fo meshfree nodes in the node
[map,parUnknownId,parUnknownVal]=generateMap(tree,dim); %% generate a map, map(parameter in current level) = parameter in meshfree node 

%%% 1D optimization (too lazy to incorporate this inside the loop for now),
%%% I have a different function for 1D and higehr Dimeniosnal optimization
maxSamp=200;
numLHSamp=2;
tic
[fOptVal(i),parOptVal]=optimization1d(parUnknownVal,parUnknownId,map,obs,simu,dim,maxSamp,numLHSamp,boundL,boundU,path_out,i);
toc
tree=updateTreeValue(tree,parUnknownId,parOptVal); %%% update the value of root node with optimum

%%% change tree structure, root node + two children
leaf = hierarchy(root).children;
tree.children(:,1) = leaf;
% child 1 :
tree.id(2) = hierarchy(leaf(1)).id; % new node id
tree.parent(2) = root; % parent of this node
tree.children(:,2) =-1; % children of this node 
tree.val(2)='u'; % paremeter value of thsi node, so far unknown
tree.mfree(2) = length(hierarchy(tree.id(2)).mfree); 
% child 2 :
tree.id(3) = hierarchy(leaf(2)).id;
tree.parent(3) = root;
tree.children(:,3) =-1;
tree.val(3)='u';
tree.mfree(3) = length(hierarchy(tree.id(3)).mfree);

clear hierarchy;clear root; clear leaf; clear path_in;

% generate map, generate initial value using coarse to fine initialization
[map,parUnknownId,parUnknownVal]=generateMap2(tree,dim,boundL,boundU); 

%% outer iteration begins
close all;
conv=0; % convergence
mergeUnknownId=[];mergeUnknownVal=[];
while(1)
    
    % step 1: optimize present leaves using GP-BO
    i=i+1;
    pdim=length(parUnknownId); % dimension of unknown parameters
    if (pdim<=6)
        maxSamp=500; % maximum number of iteration
        numLHSamp=2; % maximum number of initial samples to generate
    elseif (pdim<=10)
        maxSamp=1000;
        numLHSamp=50;
    else
        numLHSamp=100;
        maxSamp=1500;
    end    
    tic
    [fOptVal(i),parOptVal]=optimization(parUnknownVal,parUnknownId,map,obs,simu,dim,maxSamp,numLHSamp,boundL,boundU,path_out,i); % GP-BO 
    toc
    
    %%% step 2: update the values of leaves node using optimum
    tree=updateTreeValue(tree,parUnknownId,parOptVal);
    
    %%% step 3:  display result at this stage and save
    save([path_out int2str(i) '_parest_data'])
%     figure(2)
%     dispTree(tree,parTrue)
%     title('subfig (left to right, top to bottom) : true values , calculate values, hierarchy id, tree id ')
%     figure(3)
%     title('title: solution, true, error')
%     showResult(parOptVal,parUnknownId,map,corMfree,boundL,boundU,i,path_out,parTrue);
%     drawnow
    
    %%%% Step 4:  convergence criteria 1: if function value does not increase for three times
    df(i)=fOptVal(i)-fOptVal(i-1);
    if(df(i)<=10^-3)
        conv= conv+1;
    end
    if(conv==1)
        break;
    end   
    
    %%% Step 5: adaptive spatial refine and coarsening
    % calculate gain, determine nodes that need split and nodes that need retraction/merging
    [mergeIdParent,splitCidChildren,splitIdChildren,tree] = findGainlessClusterSplit(tree,obs,simu,map,fOptVal(i),mergeUnknownId,mergeUnknownVal);
    % splitting
    [tree,split] = splitCluster(tree,mergeIdParent,splitIdChildren,splitCidChildren); %%%% split nodes
    % retraction
    [tree,mergeUnknownId,mergeUnknownVal] = retractClusterSplit(tree,mergeIdParent,mergeUnknownId,mergeUnknownVal);  
    
    % convergence criteria 2: all leaves in the hierarchy so further split
    % impossible (this type of convergence never occurs in practice)
    if(split==0)
        disp('ALL LEAVES');
        break;
    end
    
    %%% Step 6: coarse to fine initialization and mapping
    [map,parUnknownId,parUnknownVal]=generateMap2(tree,dim,boundL,boundU);
    
end

