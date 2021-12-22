%%%%%%%%%%% Computation of the distance vectors and the VQA metric evaluation ##########

clc;
clear;
close all;

dmos = load('path/to/dmos');
m = load('/path/to/mssim.mat'); % load the MS-SSIM of the dataset computed using msssim.m

n1 = 10; % number of refernce videos
n2 = 16; % number of distorted videos per reference video
n3 = 15; % number of patches in a video frame
z = fspecial('gaussian', [3 5]); % Gaussian weights for the frame patches 3*5 = 15
for sub=1 : 3
    count = 0;
    l = load(['active_voxels_sub',num2str(sub),'.mat'],'k');
    vox = xlsread(['sub',num2str(sub),'_Hcor_lcc.xlsx']);
    vox = vox(1:100,:);
    [a,b] = hist(vox,unique(vox));
    id=b;
    
    d=[];
    
    for kk = 1 : n1
        for j = 1 : n2
            count=count+1;
            for ii = 1 : n3
                disp(['sub : ',num2str(sub),' patch : ',num2str(ii),' vid : ',num2str(j)]);
                
                l1= load('/path/to/the predicted/voxel response/of the reference/video');
                l2=load('/path/to/the predicted/voxel response/of the disorted/video');
                
                if ii==1
                    
                    d(count,:)=z(ii)*mean(abs(l1.data-l2.data)); % mean of absolute difference over time
                    
                else
                    d(count,:) = d(count,:)+z(ii)*mean(abs(l1.data-l2.data));
                    
                end
            end
            
        end
    end
    d=mean(d,2);  % mean of absolute difference over voxels
    d=horzcat(d,m);
    no_of_iterations = 100;
    TrainRatio = 0.8;
    TestRatio = 0.2;
    ValidationRatio = 0;
    temp=[];
    model={};
    for iter = 1:1:no_of_iterations
        [trainInd,valInd,testInd] = dividerand(size(d,1),TrainRatio,ValidationRatio,TestRatio);
        mdl = fitrsvm(d(trainInd,:),dmos(trainInd),'KernelFunction','rbf','KernelScale','auto','Standardize',true,'CacheSize','maximal');
        y_cap = predict(mdl,d(testInd,:));
        mos_cap = dmos(testInd);
        temp(iter,:) = calculatepearsoncorr(y_cap,mos_cap);
        model{iter}=mdl;
        median(temp)
    end
end


