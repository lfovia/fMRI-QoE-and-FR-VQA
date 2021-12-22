%%% Vim-2 fmri data processing
fmripath = '/path/to/store/groundtruth_fmri/sub1/';

s=load('/path/to/vim-2/VoxelResponses_subject1.mat');
data1 = transpose(s.rt);
data2 = transpose(s.rv);

% Standardization: remove the mean and divide the standard deviation
    data1 = bsxfun(@minus, data1, mean(data1,2));
    data1 = bsxfun(@rdivide, data1, std(data1,[],2));
    data1(isnan(data1)) = 0;

    data2 = bsxfun(@minus, data2, mean(data2,2));
    data2 = bsxfun(@rdivide, data2, std(data2,[],2));
    data2(isnan(data2)) = 0;

    save([fmripath,'training_fmri.mat'], 'data1', '-v7.3');
    save([fmripath,'validation_fmri.mat'], 'data2', '-v7.3');
