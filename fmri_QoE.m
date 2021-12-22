%%%%% Computation of the distance vectors and the QoE prediction %%%%%%%%%

clc;
clear;
close all;

tvsq = load('TVSQ from QoE dataset');
oq = load ('TVSQ from QoE dataset');
m = load('/path/to/mssim.mat'); % load the MS-SSIM of the dataset computed using msssim.m

n1 = 3; % number of refernce videos
n2 = 5; % number of distorted videos per reference video
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
                
                l1= load('/path/to/predicted/voxel response/of the reference/video');
                l2=load('/path/to/predicted/voxel response/of the disorted/video');
                
                if ii==1
                    e=z(ii)*(abs(l1.data-l2.data));
                    d=z(ii)*mean(abs(l1.data-l2.data));
                    
                else
                    e = e+z(ii)*(abs(l1.data-l2.data));
                    d = d+z(ii)*mean(abs(l1.data-l2.data),2);  % mean absolute differenve over voxels
                    
                end
                
                if count == 1
                    
                    err=e;
                    dist = d;
                else
                    err=vertcat(err,e);
                    dist = vertcat(dist,d);
                end
                
            end
        end
    d1=mean(dist); % mean absolute differenve over time
    oq_cor=calculatepearsoncorr(d1,OQ); % calculation of the overall QoE 
    
    dist1=horzcat(dist,m); % feature matrix for calculating the QoE with Average voxel prediction
%     dist1=horzcat(err,m); % feature matrix for calculating the QoE with individual voxel prediction

%%%%% fit an svr and compute the correlation scores using leave-one-out
%%%%% strategy and then find the average correlation value.
x1=transpose(dist1(1:3000,:));
t1=tvsq(1:3000);
x2=dist1(3001:4500,:);
t2=tvsq(3001:4500);
mdl=fitrsvm(x1',t1','KernelFunction','rbf','KernelScale','auto','Standardize',true,'CacheSize','maximal');
y=predict(mdl,x2);
r1=calculatepearsoncorr(y,t2)

x1=transpose(dist1(1501:4500,:));
t1=tvsq(1501:4500);
x2=dist1(1:1500,:);
t2=tvsq(1:1500);
mdl=fitrsvm(x1',t1','KernelFunction','rbf','KernelScale','auto','Standardize',true,'CacheSize','maximal');
y=predict(mdl,x2);
r2=calculatepearsoncorr(y,t2)

x1=transpose(cat(1,dist1(1:1500,:),dist1(3001:4500,:)));
t1=transpose(vertcat(tvsq(1:1500),tvsq(3001:4500)));
x2=(dist1(1501:3000,:));
t2=tvsq(1501:3000);
mdl=fitrsvm(x1',t1','KernelFunction','rbf','KernelScale','auto','Standardize',true,'CacheSize','maximal');
y=predict(mdl,x2);
r3=calculatepearsoncorr(y,t2)

tvsq_cor=mean([r1;r2;r3]);
end


