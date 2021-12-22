######## Extraction of features at the output of the convolutional and dense layers of the Alexnet #########
# 
#
# Reference: 
# Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). 
# Imagenet classification with deep convolutional neural networks.
# In Advances in neural information processing systems (pp. 1097-1105).
 

import numpy as np
import matplotlib.pyplot as plt
import h5py
import time

# The caffe module needs to be on the Python path;
caffe_root = 'path/to/caffe'  # this is the path to caffe, {caffe_root}/examples
data_root = '/path/to/framefolder/'
dataroot = '/path/to/feature/maps/folder/' # this is the path where the extracted feature maps are saved
import caffe
caffe.set_device(0) # if use gpu
caffe.set_mode_gpu()

######################################## Load AlexNet model ###################################################
model_def = caffe_root + 'models/bvlc_alexnet/deploy.prototxt' # load model definition
model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel' # load model weights
imsize = 227

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode 
                
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu[:,15:242,15:242]  # the mean (BGR) pixel values
    
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel


####################################### Classify example images ##############################################  
image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)

# set the size of the input
net.blobs['data'].reshape(1,        # batch size
                          3,        # 3-channel (BGR) images
                          imsize, imsize)  # image size is 227x227
net.blobs['data'].data[...] = transformed_image

# perform classification
output = net.forward()
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
print ('predicted class is:', output_prob.argmax())

# load ImageNet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'    
labels = np.loadtxt(labels_file, str, delimiter='\t')
print ('output label:', labels[output_prob.argmax()])

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print ('probabilities and labels:')
zip(output_prob[top_inds], labels[top_inds])


########################################### Define layer labels ###################################################
layer_name_list = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8']

########################################## process training movies of fmri dataset ################################################
# read middle layer feature maps
start = time.time()
srate = 15 #frame rate of the video
Ns = 120 # number of training movie segments. Divide the video into segments for ease of storage and processing.
numOfimages = srate*60*9 # number of images in each video segment
numlist = np.arange(0,numOfimages,int(15/srate)) # subsample if needed

for seg in range(Ns):
     
    foldpath = dataroot+'/AlexNet_feature_maps_seg'+ str(seg+1)+'.h5'
    store = h5py.File(foldpath,'w')
    act={}
    for lay_idx in range(0,len(layer_name_list)): 
        layer_name = layer_name_list[lay_idx]
        grp1 = store.create_group(layer_name)
        temp = net.blobs[layer_name].data.shape 
        temp = list(temp)
        temp[0]=len(numlist)
        temp = tuple(temp)
        act[lay_idx]=grp1.create_dataset('data', temp, dtype='float16')
    k = 0
    for im in numlist:
        image = caffe.io.load_image(data_root+'/frames/im-'+str(im+1)+'.jpg')
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        for lay_idx in range(0,len(layer_name_list)): 
            layer_name = layer_name_list[lay_idx]
            act[lay_idx][k,:] = net.blobs[layer_name].data
        k = k + 1    
    store.close() 

endtime = time.time()
    
print("--- %s seconds ---" % (endtime - start))

########################################## process testing movies of fmri dataset ################################################
# read middle layer feature maps
start = time.time()
srate = 15
Ns = 9 # number of testing movie segments
numOfimages = srate*60*9 # number of images in each video segment
numlist = np.arange(0,numOfimages,int(15/srate)) # subsample if needed

for seg in range(Ns):
     
    foldpath = dataroot+'/AlexNet_feature_maps_val'+ str(seg+1)+'.h5'
    store = h5py.File(foldpath,'w')
    act={}
    for lay_idx in range(0,len(layer_name_list)): 
        layer_name = layer_name_list[lay_idx]
        grp1 = store.create_group(layer_name)
        temp = net.blobs[layer_name].data.shape 
        temp = list(temp)
        temp[0]=len(numlist)
        temp = tuple(temp)
        act[lay_idx]=grp1.create_dataset('data', temp, dtype='float16')
    k = 0
    for im in numlist:
        image = caffe.io.load_image(data_root+'/frames/im-'+str(im+1)+'.jpg')
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        for lay_idx in range(0,len(layer_name_list)): 
            layer_name = layer_name_list[lay_idx]
            act[lay_idx][k,:] = net.blobs[layer_name].data
        k = k + 1    
    store.close() 

endtime = time.time()
    
print("--- %s seconds ---" % (endtime - start)) 



########################################## process movies for VQA using parallel processing ################################################

'''# read middle layer feature maps
#start = time.time()
srate = 30

c = 0

def my_function(vid):
  t = os.listdir("/path/to/dataset/frames/"+vid);
  numOfimages = len(t)
  numlist = np.arange(0,numOfimages,1) # subsample if needed  

  for i in range (14):
    foldpath = dataroot+str(i+1)+'/'+vid.replace('.mat','_fmaps.h5')
    store = h5py.File(foldpath,'w')
    act={}
    for lay_idx in range(0,len(layer_name_list)): 
        layer_name = layer_name_list[lay_idx]
        grp1 = store.create_group(layer_name)
        temp = net.blobs[layer_name].data.shape 
        temp = list(temp)
        temp[0]=len(numlist)
        temp = tuple(temp)
        act[lay_idx]=grp1.create_dataset('data', temp, dtype='float16')
    k = 0
    for im in range(len(t)):
        print ("frame %d, patch %d " %(im+1,i+1))
        img1 = caffe.io.load_image(data_root+'frames/'+vid+'/im-'+str(im+1)+'.png')#'_'+str(l+1)+     patch %d,i+1
        img1 = np.swapaxes(img1,0,2)
        ####### Divide the image into patches of size 227x227 ############
        if i==0:
		image = img1[0:226,0:226,:]
	elif i==1:
		image = img1[0:226,263:489,:]
	elif i==2:
		image = img1[0:226,526:752,:]
	elif i==3:
		image = img1[0:226,789:1015,:]
	elif i==4:
		image = img1[0:226,1052:1278,:]
	elif i==5:
		image = img1[247:473,0:226,:]
	elif i==6:
		image = img1[247:473,263:489,:]
	elif i==7:
		image = img1[247:473,526:752,:]
	elif i==8:
		image = img1[247:473,789:1015,:]
	elif i==9:
		image = img1[247:473,1052:1278,:]
	elif i==10:
		image = img1[493:719,0:226,:]
	elif i==11:
		image = img1[493:719,263:489,:]
	elif i==12:
		image = img1[493:719,526:752,:]
	elif i==13:
		image = img1[493:719,789:1015,:]
	elif i==14:
		image = img1[493:719,1052:1278,:]
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        for lay_idx in range(0,len(layer_name_list)): 
            layer_name = layer_name_list[lay_idx]
            act[lay_idx][k,:] = net.blobs[layer_name].data
        k = k + 1    
    store.close() 
    
inputs = os.listdir("/path/to/dataset/frames")
pool = mp.Pool(mp.cpu_count())      
results = pool.map(my_function, [i for i in inputs])
pool.close()   
print("--- %s seconds ---" % (endtime - start))'''
