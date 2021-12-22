%%%%%%%% Processing the CNN features extracted from videos for dimensionality reduction using PCA %%%%%%%%%%%
% 
%%%%%%%%%%%%%%%%%%%%  References %%%%%%%%%%%%%%%%%%% 
% Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
% and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
% Cortex, In press.
%
% Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). 
% Imagenet classification with deep convolutional neural networks.
% In Advances in neural information processing systems (pp. 1097-1105).
%
% Zha, H., & Simon, H. D. (1999). 
% On updating problems in latent semantic indexing. 
% SIAM Journal on Scientific Computing, 21(2), 782-791.
% 
% Zhao, H., Yuen, P. C., & Kwok, J. T. (2006). 
% A novel incremental principal component analysis and its application for face recognition. 
% IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 36(4), 873-886.
%
% Wen, H., Shi, J., Chen, W., & Liu, Z. (2017). 
% Transferring and Generalizing Deep-Learning-based Neural Encoding Models across Subjects. 
% bioRxiv, 171017.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calculate the temporal mean and standard deviation of the feature time series of CNN units
%CNN layer labels
layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};
dataroot1 = '/path/to/fmaps/'; % path to the feature maps folder
data_root = '/path/to/avg_maps/'; % path to save the PCA reduced feature maps

%%%%% calculate the temporal mean 

for lay = 1 : length(layername)
    N = 0;
    for seg = 1 : 120 % number of video segments or videos in a dataset
        disp(['Layer: ',layername{lay},'; Seg: ', num2str(seg)]);
        secpath = [dataroot1,'AlexNet_feature_maps_seg_', num2str(seg),'.h5'];
        if exist(secpath,'file')
            lay_feat = h5read(secpath,[layername{lay},'/data']);
            dim = size(lay_feat); % convolutional layers: #kernel*fmsize1*fmsize2*#frames
            if seg == 1 
                lay_feat_mean = zeros([dim(1:end-1),1]);
            end
            lay_feat_mean = lay_feat_mean + sum(lay_feat,length(dim));
            N = N  + dim(end);
        end
    end
    lay_feat_mean = lay_feat_mean/N;
     
    h5create([dataroot,'AlexNet_feature_maps_avg.h5'],[layername{lay},'/data'],...
        [size(lay_feat_mean)],'Datatype','single');
    h5write([dataroot,'AlexNet_feature_maps_avg.h5'], [layername{lay},'/data'], lay_feat_mean);
end

%%%%% calculate the temporal standard deviation

for lay = 1 : length(layername)
    lay_feat_mean = h5read([dataroot,'AlexNet_feature_maps_avg.h5'], [layername{lay},'/data']);
    N = 0;
    for seg = 1 : 120
        disp(['Layer: ',layername{lay},'; Seg: ', num2str(seg)]);
        secpath = [dataroot1,'AlexNet_feature_maps_seg_', num2str(seg),'.h5'];
        if exist(secpath,'file')
            lay_feat = h5read(secpath,[layername{lay},'/data']);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = lay_feat.^2;
            dim = size(lay_feat);
            if seg == 1 
                lay_feat_std = zeros([dim(1:end-1),1]);
            end
            lay_feat_std = lay_feat_std + sum(lay_feat,length(dim));
            N = N  + dim(end);
        end
    end
    lay_feat_std = sqrt(lay_feat_std/(N-1));
    lay_feat_std(lay_feat_std==0) = 1;
    h5create([dataroot,'AlexNet_feature_maps_std.h5'],[layername{lay},'/data'],...
        [size(lay_feat_mean)],'Datatype','single');
    h5write([dataroot,'AlexNet_feature_maps_std.h5'], [layername{lay},'/data'], lay_feat_std);
end

%% Reduce the dimension of the AlexNet features for encoding models
% Dimension reduction by using principal component analysis (PCA)
% Here provides two ways to compute the PCAs. If the memory is big enough
% to directly calculate the svd, then use the method 1, otherwise use the
% SVD-updating algorithm (Zha and Simon, 1999; Zhao et al., 2006; Wen et al. 2017).
% 
% % % % % % % % % Direct SVD % % % % % % % % % % % % % %
% This requires big memory to compute. It's better to reduce the
% frame rate of videos or use the SVD-updating algorithm.

for lay = 1 : length(layername)
    lay_feat_mean = h5read([dataroot,'AlexNet_feature_maps_avg.h5'], [layername{lay},'/data']);
    lay_feat_std = h5read([dataroot,'AlexNet_feature_maps_std.h5'], [layername{lay},'/data']);
    lay_feat_mean = lay_feat_mean(:);
    lay_feat_std = lay_feat_std(:);
    
    % Concatenating the feature maps across training movie segments. Ensure that 
    % the memory is big enough to concatenate all movie segments. Otherwise, 
    % use the SVD-updating algorithm. 
    
    for seg = 1 : 18
        disp(['Layer: ',layername{lay},'; Seg: ', num2str(seg)]);
        secpath = [dataroot,'AlexNet_feature_maps_seg', num2str(seg),'.h5'];
            
        lay_feat = h5read(secpath,[layername{lay},'/data']);
        dim = size(lay_feat);
        Nu = prod(dim(1:end-1)); % number of units
        Nf = dim(end); % number of frames
        lay_feat = reshape(lay_feat,Nu,Nf);% Nu*Nf
        
%         lay_feat = lay_feat(:,1:3:end); % downsample if encounter memory issue
%         Nf = size(lay_feat,2);
        
        if seg == 1
            lay_feat_cont = zeros(Nu, Nf*18,'single');
        end
        lay_feat_cont(:, (seg-1)*Nf+1:seg*Nf) = lay_feat;
    end
    
    % standardize the time series for each unit  
    lay_feat_cont = bsxfun(@minus, lay_feat_cont, lay_feat_mean);
    lay_feat_cont = bsxfun(@rdivide, lay_feat_cont, lay_feat_std);
    lay_feat_cont(isnan(lay_feat_cont)) = 0; % assign 0 to nan values
    
    %[B, S] = svd(lay_feat_cont,0);
    if size(lay_feat_cont,1) > size(lay_feat_cont,2)
        R = lay_feat_cont'*lay_feat_cont/size(lay_feat_cont,1);
        [U,S] = svd(R);
        s = diag(S);
        
        % keep 99% variance
        ratio = cumsum(s)/sum(s); 
        Nc = find(ratio>0.99,true,'first'); % number of components
        
        S_2 = diag(1./sqrt(s(1:Nc))); % S.^(-1/2)
        B = lay_feat_cont*(U(:,1:Nc)*S_2/sqrt(size(lay_feat_cont,1)));
        % I = B'*B; % check if I is an indentity matrix
        
    else
        R = lay_feat_cont*lay_feat_cont';
        [U,S] = svd(R);
        s = diag(S);
        
        % keep 99% variance
        ratio = cumsum(s)/sum(s); 
        Nc = find(ratio>0.99,true,'first'); % number of components
        
        B = U(:,1:Nc);
    end

    % save principal components
    save([dataroot,'svd/AlexNet_feature_maps_pca_layer',num2str(lay),'.mat'], 'B', 's', '-v7.3');
end

% % % % % % % % % SVD-updating algorithm % % % % % % % % %
Niter = 2; % number of iteration to compute principle component

for lay = 1 : length(layername)
    lay_feat_mean = h5read([dataroot,'AlexNet_feature_maps_avg.h5'], [layername{lay},'/data']);
    lay_feat_std = h5read([dataroot,'AlexNet_feature_maps_std.h5'], [layername{lay},'/data']);
    
    k0 = 0;
    percp = 0.99; % explain 99% of the variance of every movie segments
    for iter =  1  : Niter
        for seg = 1 : 120
            disp(['Layer: ', num2str(lay),'; Seg: ',num2str(seg),'; Comp:',num2str(k0)]);
            secpath = [dataroot1,'AlexNet_feature_maps_seg_', num2str(seg),'.h5'];
            if exist(secpath,'file')==2
                lay_feat = h5read(secpath,[layername{lay},'/data']);
                lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
                lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
                lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
                
                dim = size(lay_feat);
                lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
                if (seg == 1) && (iter == 1) 
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
                else
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
                end  
            else
               disp('file doesn''t exist'); 
            end
        end
    end
    s = diag(S);
    
    % save principal components
    save([dataroot,'svd/AlexNet_feature_maps_svd_layer',num2str(lay),'.mat'], 'B', 's', '-v7.3');
end


% % % Process the videos for dimensionality reduction
% % CNN layer labels
layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};
dataroot1 = '/path/to/fmaps/';
dataroot = '/path/to/avg_maps/';

% % The sampling rate should be equal to the sampling rate of CNN feature
% % maps. If the CNN extracts the feature maps from movie frames with 30
% % frames/second, then srate = 30.

% srate = 15; 
 
%%% Here is an example of using pre-defined hemodynamic response function
%%% (HRF) with positive peak at 4s.
% p  = [5, 16, 1, 1, 6, 0, 32];
% hrf = spm_hrf(1/srate,p);
% hrf = hrf(:);
% figure; plot(0:1/srate:p(7),hrf);

% %Dimension reduction for CNN features
for lay = 1 : length(layername)
    lay_feat_mean = h5read([dataroot,'AlexNet_feature_maps_avg.h5'], [layername{lay},'/data']);
    lay_feat_std = h5read([dataroot,'AlexNet_feature_maps_std.h5'], [layername{lay},'/data']);
    load([dataroot,'svd/AlexNet_feature_maps_svd_layer',num2str(lay),'.mat'], 'B');


      
    % Dimension reduction for training data
    for seg = 1 : 120
        disp(['Layer: ', num2str(lay),'; Seg: ',num2str(seg)]);
        secpath = [dataroot1,'test2_',num2str(i),'/AlexNet_feature_maps_seg', num2str(seg),'.h5'];
        if exist(secpath,'file')==2
            lay_feat = h5read(secpath,[layername{lay},'/data']);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
            
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
            Y = lay_feat'*B/sqrt(size(B,1)); % Y: #time-by-#components
%            ts = conv2(hrf,Y); % convolude with hrf
%            ts = ts(4*srate+1:4*srate+size(Y,1),:);
%            ts = ts(8:srate:end,:); % downsampling
             
            h5create([dataroot,'/AlexNet_feature_maps_pcareduced_seg', num2str(seg),'.h5'],[layername{lay},'/data'],...
                [size(Y)],'Datatype','single');
            h5write([dataroot,'/AlexNet_feature_maps_pcareduced_seg', num2str(seg),'.h5'], [layername{lay},'/data'], Y);
        end
    end
%%%%%%%%%% While processing the videos for quality assessment only this section 
%%%%%%%%%% can be used and convolution with the hrf function can be uncommented
%%%%%%%%%% modify the code accordingly (save the feature 'ts' instead of 'Y')  
%%%%%%%%%% Everything below this line can be commented

    %%%%%  Dimension reduction for testing data
    for seg = 1 :9
        disp(['Layer: ', num2str(lay),'; Test: ',num2str(seg)]);
        secpath = [dataroot1,'AlexNet_feature_maps_val_', num2str(seg),'.h5'];
        if exist(secpath,'file')==2
            lay_feat = h5read(secpath,[layername{lay},'/data']);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
            
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
            Y = lay_feat'*B/sqrt(size(B,1)); % Y: #time-by-#components
                 
            h5create([dataroot,'AlexNet_feature_maps_pcareduced_val_', num2str(seg),'.h5'],[layername{lay},'/data'],...
                [size(Y)],'Datatype','single');
            h5write([dataroot,'AlexNet_feature_maps_pcareduced_val_', num2str(seg),'.h5'], [layername{lay},'/data'], Y);
        end
    end  
     
end


%% Concatenate the dimension-reduced CNN features of training movies
% % CNN layer labels
layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};
dataroot = '/media/sailaja/MNS_HDD/vim/avg_maps/';
% % The sampling rate should be equal to the sampling rate of CNN feature
% % maps. If the CNN extracts the feature maps from movie frames with 30
% % frames/second, then srate = 30. It's better to set srate as even number
% % for easy downsampling to match the sampling rate of fmri (2Hz).

srate = 15; 
 
% Here is an example of using pre-defined hemodynamic response function
% (HRF) with positive peak at 4s.
p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);
%figure; plot(0:1/srate:p(7),hrf);
%     
for lay = 1 : length(layername)
    for seg = 1:120
        disp(['Layer: ', num2str(lay),'; Seg: ',num2str(seg)]);
        secpath = [dataroot,'/AlexNet_feature_maps_pcareduced_seg', num2str(seg),'.h5'];      
        lay_feat = h5read(secpath,[layername{lay},'/data']);% #time-by-#components
        dim = size(lay_feat);
        Nf = dim(1); % number of frames
        if seg == 1
           Y = zeros([Nf*10, dim(2)],'single'); 
        end
        Y((seg-1)*Nf+1:seg*Nf, :) = lay_feat;
            
            ts = conv2(hrf,Y); % convolude with hrf
            ts = ts(4*srate+1:4*srate+size(Y,1),:);
            ts = ts(8:srate:end,:); % downsampling
                  
    end
    h5create([dataroot,'AlexNet_feature_maps_pcareduced_concatenated.h5'],[layername{lay},'/data'],...
        [size(ts)],'Datatype','single');
    h5write([dataroot,'AlexNet_feature_maps_pcareduced_concatenated.h5'], [layername{lay},'/data'], ts);
end

%% Concatenate the dimension-reduced CNN features of test movies
% % CNN layer labels
layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/fc6';'/fc7';'/fc8'};
dataroot = '/path/to/avg_maps/';
% % The sampling rate should be equal to the sampling rate of CNN feature
% % maps. If the CNN extracts the feature maps from movie frames with 30
% % frames/second, then srate = 30. It's better to set srate as even number
% % for easy downsampling to match the sampling rate of fmri (2Hz).

srate = 15; 
 
% Here is an example of using pre-defined hemodynamic response function
% (HRF) with positive peak at 4s.
p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);
%figure; plot(0:1/srate:p(7),hrf);
%     
for lay = 1 : length(layername)
    for seg = 1:9
        disp(['Layer: ', num2str(lay),'; Seg: ',num2str(seg)]);
        secpath = [dataroot,'/AlexNet_feature_maps_pcareduced_val', num2str(seg),'.h5'];      
        lay_feat = h5read(secpath,[layername{lay},'/data']);% #time-by-#components
        dim = size(lay_feat);
        Nf = dim(1); % number of frames
        if seg == 1
           Y = zeros([Nf*10, dim(2)],'single'); 
        end
        Y((seg-1)*Nf+1:seg*Nf, :) = lay_feat;
            
            ts = conv2(hrf,Y); % convolude with hrf
            ts = ts(4*srate+1:4*srate+size(Y,1),:);
            ts = ts(8:srate:end,:); % downsampling
                  
    end
    h5create([dataroot,'AlexNet_feature_maps_pcareduced_concatenated_val.h5'],[layername{lay},'/data'],...
        [size(ts)],'Datatype','single');
    h5write([dataroot,'AlexNet_feature_maps_pcareduced_concatenated_val.h5'], [layername{lay},'/data'], ts);
end

