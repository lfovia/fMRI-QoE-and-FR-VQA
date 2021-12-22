
%% This code is intended to predict the voxel responses from the encoded models for the videos in the quality assessment dataset 

clc;
clear ;
close all;
for  sub=1:3
    data_root = '/path/to/PCA-reduced/feature-maps/';
    a=dir([data_root,'/*.h5']);
    len = length(a);
 parfor j=1:len

     l1=load(['active_voxels_sub',num2str(sub),'.mat'],'k');
     dataroot = ['Encode/sub',num2str(sub),'/'];
     layername = {'/conv1';'/conv2';'/conv3a';'/conv3b';'/conv4a';'/conv4b';'/conv5a';'/conv5b';'/fc6';'/fc7';'/fc8'};%{'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};
     l2=load([dataroot, 'cross_corr_fmri_cnn.mat'],'layidx');
             layidx = l2.layidx(l1.k);
     Y = h5read([data_root,a(j).name],[layername{11},'/data']);
     X = zeros(size(Y,1),length(l1.k));
     for lay=1:length(layername)
         f=find(layidx==lay);
         l3=load([dataroot,'encoding_models_layer',num2str(lay),'.mat'], 'W');
         W=l3.W(:,l1.k);
         W = W(:,f);
         Y = h5read([data_root,a(j).name],[layername{lay},'/data']);

         disp(['vid:',num2str(j),' layer: ', num2str(lay)]);%,' patch:',num2str(ii)]);
         X(:,f) = Y*W;
         
     end
     m=matfile(sprintf(['/path/to/save/responses/sub',num2str(sub),'/',strrep(a(j).name,'_avgmaps.h5','.mat')]),'writable',true);
         m.X=X;
 end
end

