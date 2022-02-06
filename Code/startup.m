function startup
disp(['1. adding path for hierarchy scripts ...']);
%addpath(genpath(pwd()))
cd hierarchy/
addpath(genpath(pwd()));
cd ../
disp(['2. adding path for model scripts ...']);
cd model/
addpath(genpath(pwd()));
cd ../
disp(['3. adding path for optimization scripts ...']);
cd GP_BO_optimization/
addpath(genpath(pwd()));
cd ../
disp(['4. adding path for showResult scripts ...']);
cd resultVisu/
addpath(genpath(pwd()));
cd ../
disp(['4. adding path for tree scripts ...']);
cd tree/
addpath(genpath(pwd()));
cd ../
disp(['4. adding path for initialization scripts ...']);
cd initialization/
addpath(genpath(pwd()));
cd ../
disp([' completed ....']);
end