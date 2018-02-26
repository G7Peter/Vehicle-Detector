#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import pickle
import math
from skimage.feature import hog

#Test reading in an image
image = mpimg.imread('test_images/test1.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Dataset Analysis

non_vehicles = glob.glob('non-vehicles/**/*.png')
vehicles = glob.glob('vehicles/**/*.png')
print(len(vehicles),len(non_vehicles))


# ## Helper Functions

# Below are some helper functions. They should look familiar from the lesson!

#image color space convertion
def convert_color(img, cspace='RGB'):
    if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    return feature_image
    
#hog features detection. This method was duplicated from lesson materials
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
        return features

#getting feature list
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
#getting color histogram
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

#extract features method. The code for this method was mostly duplicated from course lesson material.
def extract_features(imgs, cspace='RGB', orient = 9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel = 'ALL'):
    # Create a list to append feature vectors to
    features = []
    i=0
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        print(file)
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                feature_image = feature_image.astype(np.float32)/255
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                feature_image = feature_image.astype(np.float32)/255
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                feature_image = feature_image.astype(np.float32)/255
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                feature_image = feature_image.astype(np.float32)/255
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
                feature_image = feature_image.astype(np.float32)/255
        else: feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
        
    # Return list of feature vectors
    return features

# ## Creating Dataset for Classifier

start = time.time()
print(start)
v_features = extract_features(vehicles, cspace=cspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
notv_features = extract_features(non_vehicles, cspace=cspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

end = time.time()
print(round(end-start, 2), 'Seconds to extract features...')

# Create an array stack of feature vectors for car and noncar images
X = np.vstack((v_features, notv_features)).astype(np.float64)
print(len(X))  

X_scale = StandardScaler().fit(X)
# Apply the scaler to X
X_scaled = X_scale.transform(X)
print(len(X_scaled))

# Labels vector definition
y = np.hstack((np.ones(len(v_features)), np.zeros(len(notv_features))))
print(len(y))

# Split up data into randomized training and test sets
rand = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=rand)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Save the data for easy access
pickle_file = 'Dataset.p'
print('Saving data to pickle file...')
try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {
                'cspace': cspace,
                'orient' : orient,
                'pix_per_cell': pix_per_cell,
                'cell_per_block' : cell_per_block,
                'hog_channel' : hog_channel,
                'scaler' : X_scale,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test                
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
    
print('Datasets cached in pickle file.')

