%% Training voxel-wise encoding models

for sub=1%:3
load(['/media/sailaja/MNS_HDD/vim/sub',num2str(sub),'/training_fmri.mat'],'data1'); % from movie_fmri_processing.m
load(['/media/sailaja/MNS_HDD/vim/active_voxels_sub',num2str(sub),'.mat'],'k');
dataroot =/path/to/save/encoding/models;%['/media/sailaja/MNS_HDD/vim/Encode/sub',num2str(sub),'/'];
X = transpose(data1);
X=X(:,k);
lambda = (0.1:0.1:1);
nfold = 9;
layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};

for lay = 1 : length(layername)
        Y = h5read(('/media/sailaja/MNS_HDD/vim/avg_maps/AlexNet_feature_maps_pcareduced_concatenated.h5'),[layername{lay},'/data']);% #time-by-#components
        
        [W, Rmat, Lambda] = voxelwise_encoding(Y, X, lambda, nfold);
        save([dataroot,'encoding_models_layer',num2str(lay),'.mat'], 'W', 'Rmat', 'Lambda');

         
end


%% Map hierarchical CNN features to brain
% Correlating all the voxels to all the CNN units is time-comsuming.

voxels = k; 
for lay = 1 : length(layername)
    lay_feat_concatenated = h5read(('/media/sailaja/MNS_HDD/vim/avg_maps/AlexNet_feature_maps_pcareduced_concatenated.h5'),[layername{lay},'/data']);% #time-by-#components
    dim = size(lay_feat_concatenated);
    Nu = dim(2);% number of units
    Nf = dim(1); % number of time points
    lay_feat_concatenated = transpose (lay_feat_concatenated)';

    Rmat = zeros([Nu,length(voxels)]);
    k1 = 1;
    while k1 <= length(voxels)
       disp(['Layer: ',num2str(lay),'; Voxel: ', num2str(k1)]);
       k2 = min(length(voxels), k1+100);
       R = amri_sig_corr(lay_feat_concatenated, fmri_avg(voxels(k1:k2),:)');
       Rmat(:,k1:k2) = R;
       k1 = k2+1;
    end
    save([dataroot, 'cross_corr_fmri_cnn_layer',num2str(lay),'.mat'], 'Rmat', '-v7.3');
end

lay_corr = zeros(length(layername),length(voxels));
for lay = 1 : length(layername)
    disp(['Layer: ',num2str(lay)]);
    load([dataroot, 'cross_corr_fmri_cnn_layer',num2str(lay),'.mat'],'Rmat');
    lay_corr(lay,:) = max(Rmat,[],1);
end

% Assign layer index to each voxel
[~,layidx] = max(lay_corr,[],1);
save([dataroot, 'cross_corr_fmri_cnn.mat'],'layidx');
end

%% Display correlation profile for example voxels
%figure(100);
%for v = 1 : length(voxels)
%    plot(1:length(layername),lay_corr(:,v)','--o');
%    title(v);
%    pause;
%end
