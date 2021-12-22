%% This code is for processing the video data
% 
% 
%% Extract the movie frames from video files
% % Though the image size in Krizhevsky et al. 2012 is 224*224, the input to 
% % the AlexNet Caffe model is images with resolution 227*227.
% % Note that the framerate of the video is 15 frames/second.

imgsize = '227';
 
videofolder = '/path/to/videofolder';
framefolder = '/path/to/framefolder';
 
% % training videos
for seg = 1 : 120 % number of video segments
     videoname = 'Vstimuli';
     impath = [framefolder, '/'];
     mkdir(impath);
 
     %ffmpeg -i "/path/to/stimuli/seg1.mp4" -vf scale=227:227 "/path/to/stimuli/frames/seg1/im-%d.jpg"
     cmd = ['ffmpeg -i "',videofolder, videoname,'.avi" -vf scale=',imgsize,':',imgsize,' "',impath,'im-%d.jpg"'];
     system(cmd);
end
% % 
% 
% % Use the python code 'AlexNet_feature_extraction.py' to extract the hierachical feature maps from the video frames.


