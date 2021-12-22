%%%%%%%% Sample code to compute MS-SSIM of a video

clc;
clear;
close all;

   
list = video_names; % list of video names in the data-set
n1 = 10 ; % number of reference videos
n2 = 16 ; % number of distorted videos
 mss=[];
 c=0;
 for i=1:n1
     m1=ref_video(i); % read the i_th reference video using VideoReader or load as a mat file
     s1 = size(m1);
     for j=1:n2
         m2=dist_video(i); % read the j_th distorted video corresponding to i_th reference video
         s2 = size(m2);
         mss1=[];
         for ii=1:min(s1(1),s2(1)) % number of frames of reference or the distorted video whichever is minimum 
             im1=m1(ii,:,:,:);     %  read frames of reference video
             im1=rgb2gray(reshape(im1,[s1(2),s1(3),s1(4)]));
             im2=m2(ii,:,:,:);     %  read frames of distorted video
             im2=rgb2gray(reshape(im2,[s1(2),s1(3),s1(4)]));
             mss1(ii)=msssim(im2,im1);
         end
         mss(c)=mean(mss1);
     end
 end
 
 save('mssim.mat','mss');
