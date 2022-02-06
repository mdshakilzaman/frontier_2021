function readFiles(path)
%read mfree coordinates
% filename = strcat(path,'Measurement_Noisy.bin');
% fid = fopen(filename,'rb');
% obs = fread(fid,[370,inf],'double')';
% fclose(fid);

%read mfree coordinates
filename = strcat(path,'heart.cor');
fid = fopen(filename,'rb');
corMfree = fread(fid,[3,inf],'double')';
fclose(fid);

% initialize the dimension
dim = size(corMfree,1);

% read trans_state
filename = strcat(path,'Trans_state.bin');
fid = fopen(filename,'rb');
s = fread(fid,[dim,inf],'double');
fclose(fid);

%read Trans
filename = strcat(path,'Trans.bin');
fid = fopen(filename,'rb');
H = fread(fid,[370,inf],'double');
fclose(fid);

%Read excitation
filename = strcat(path,'heart.exc');
fid = fopen(filename,'rb');
source = fread(fid,'int');
fclose(fid);

%read left right map
filename = strcat(path,'heart_LRV.map');
fid = fopen(filename,'rb');
lrvMap = fread(fid,[dim,inf],'double');
fclose(fid);

%read seventeen segment map
filename = strcat(path,'heart.seg');
fid = fopen(filename,'rb');
ahaMap = fread(fid,[dim,inf],'int32');
fclose(fid);


% read bsp conversion map
filename = strcat(path,'elnodes_123_370.loc');
fid = fopen(filename,'rb');
maskidx = fread(fid,[1,inf],'int32');
fclose(fid);

simu.maskidx=maskidx;
% maskidx=maskidx(4:end);
% mask = zeros(370,1);
% mask(maskidx)=1;
% mask=repmat(mask,1,dim);

% initialize the stimulus locations
X0 = zeros(2*dim,1);
source = source(2:end);
X0(source == 1 | source == 3 | source ==2 | source == 4 ) = 0.5;
s(X0 == 0.5,:) = 0;

% read information for total variation
% load(strcat(path,'gpts2.mat'));
% Reg = zeros(dim*dim,1);
% gpts_num = length(gpts);
% for i = 1 : gpts_num
%     gpts1=(gpts{i}.wei).*(gpts{i}.dTd);
%     id_dsha=gpts{i}.idx_dTd;
%     id_dsha=id_dsha(:);
%     gpts12=gpts1(:);
%     Reg(id_dsha)= Reg(id_dsha)+ gpts12;
%     
% end
% dwd=reshape(Reg,dim,dim);

% save everything into a matlab file
simu.X0=X0;
simu.s=s;
%simu.H=H.*mask;
simu.H=H;
simu.dim=dim;
save('modelFiles','simu','lrvMap','ahaMap','corMfree')
clear all