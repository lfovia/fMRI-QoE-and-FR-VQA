# fMRI-QoE-and-FR-VQA
Evaluation of Quality of Experience and Full reference Video Quality Assessment using fMRI voxel models of Alex-Net

This is the implementation of our ICASSP 2021 paper VIDEO QUALITY PREDICTION USING VOXEL-WISE MODELS OF THE VISUAL CORTEX. In this repository, we provide code for training the voxel models and testing them for the video quality prediction.

If you find this work useful in your research, please cite the following paper

N. S. Mahankali and S. S. Channappayya, "Video Quality Prediction Using Voxel-Wise fMRI Models of the Visual Cortex," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 2125-2129, doi: 10.1109/ICASSP39728.2021.9414328.

   
The fMRI data-set can be downloaded from  https://crcns.org/data-sets/vc/vim-2

Data processing steps:
 
 Split the training and test videos into segments and then extract the frames of these segments using "video_preprocessing.m"
 
 The ground truth fMRI data can be processed using  "fmri_data_processing.m"
 
 Features are extracted using "AlexNet_feature_extraction.py"
 
 These features are processed for dimensionality reduction using "AlexNet_feature_processing_encoding.m"
 
 Using these dimensionality reduced features encode the voxel for all the three subjects using "encode.m"
 
 You can predict the voxel responses of new videos using "predict_response.m"
    
 The quality scores can be computed using "fmri_QoE.m" and "fMRI_VQA.m"
