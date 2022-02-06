function [mea_n,sigPow,noiPow_SD2] = genIniNoi(mea,SNR,type,PARA,flag)
% add noise to the signal
% apply the random generation function in matlab toolbox
% noise = dim1*dim2
% flag = 0, time-invariant noise; flag = 1, spatial noise
% type: noise type, if flag = 0, does not matter

[dimz,t] = size(mea);
if(SNR==0)
    fid = fopen('Measurement_noisy.bin','wb');
    fwrite(fid,mea,'double');
    fclose(fid);
    fid = fopen('Noise_mea.bin','wb');
    fwrite(fid,zeros(t,1),'double');
    fclose(fid);
elseif flag==0   % noise differes at each node, but should time invariant statistics
    sigPow = sum(mea.*mea,2)/t;
    noiPow_mean = sigPow./10^(SNR/10);
    noiPow_SD = sqrt(noiPow_mean);
    for i=1:dimz
        noise(i,:) = normrnd(0,noiPow_SD(i),1,t);
    end
    noiPow_SD2 = noiPow_mean;
else           % noise differes at each time instant, but should statistically invariant for each node at each time instant
    sigPow = sum(sum(mea.*mea))/(dimz*t)
    noiPow_mean = sigPow./10^(SNR/10)
    noiPow_SD = noiPow_mean + noiPow_mean/100*randn(1,t);
    if type== 'Gau'
		mean = 0; var = 1;
		noise = normrnd( mean, var, dimz, t);
    elseif type== 'Poi'
		[mean,var] = poisstat(PARA)
		noise = poissrnd(PARA,dimz,t);
    elseif type=='Exp'
		[mean,var] = expstat(PARA);
		noise = exprnd(PARA,dimz,t);
    elseif type == 'Ray'
		[mean,var] = raylstat(PARA);
		noise = raylrnd(PARA,dimz,t);
    elseif  type =='Uni'
		[mean,var] = unifstat(0,1);
		noise = unifrnd(0,1,dimz,t);
    else type == 'Geo'
		[mean,var] = geostat(PARA);
		noise = geornd(PARA,dimz,t);
    end
  	%normalize-- to mean=0,var=1
	noise = (noise - mean)/sqrt(var);
    % changing to var = noise_power
   noise = (ones(dimz,1)*sqrt(noiPow_SD)).*noise;
end 
    mea_n = mea + noise;
    
   
end
