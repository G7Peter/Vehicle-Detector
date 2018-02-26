#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
import time
from moviepy.editor import VideoFileClip
import pickle
import math
import numpy as np
import cv2
from skimage.feature import hog

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

#extract features method. The code for this method was mostly duplicated from course lesson material.
def extract_features(imgs, cspace='RGB', orient = 9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel = 'ALL'):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
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

# Load the dataset
data_file = 'Dataset.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)
    
cspace = data['cspace']
orient = data['orient']
pix_per_cell = data['pix_per_cell']
cell_per_block = data['cell_per_block']
hog_channel = data['hog_channel']
scaler = data['scaler']
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# ## Training Classifier

from sklearn.svm import SVC #LinearSVC

# Use a linear SVC 
#svc = LinearSVC(C=10000)
svc = SVC(C=1, kernel="linear", gamma='auto') 

# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')


# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('Current SVC predicts:     ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()

print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

#input("Press Enter to continue...")
# ## Build a Vehicle Tracking Pipeline
# 
# Below is the main function for finding cars in an image. This code is adapted from lesson material

def find_cars(img, ystart, ystop, xstart, xstop, scale, cspace, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel = 'ALL', show_all_rectangles=False):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    
    rectangles = []
    
    if cspace != 'RGB':
            if cspace == 'HSV':
                ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
                ctrans_tosearch = ctrans_tosearch.astype(np.float32)/255
            elif cspace == 'LUV':
                ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
                ctrans_tosearch = ctrans_tosearch.astype(np.float32)/255
            elif cspace == 'HLS':
                ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
                ctrans_tosearch = ctrans_tosearch.astype(np.float32)/255
            elif cspace == 'YUV':
                ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
                ctrans_tosearch = ctrans_tosearch.astype(np.float32)/255
            elif cspace == 'YCrCb':
                ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
                ctrans_tosearch = ctrans_tosearch.astype(np.float32)/255
    else: ctrans_tosearch = np.copy(img_tosearch)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    # select colorspace channel for HOG 
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else: 
        ch1 = ctrans_tosearch[:,:,hog_channel]
    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
       
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            #spatial_features = bin_spatial(subimg, size=spatial_size)
            #hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack(hog_features))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 or show_all_rectangles == True:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart)))
                
    return rectangles

# In[21]:

# find cars on a test image
#ystart = 400
#ystop = 656
#scale = 1.5
colorspace = cspace # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

# Here is your draw_boxes function 
def draw_boxes(img, bboxes, color=(0, 255, 0), thick=5):
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, rects):
        
    for bbox in rects:
        # Draw the box on the image
        #cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 3)
        
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 4)
    # Return the image and final rectangles
    return img

def calculate_labeled_boxes(img, labels):
    # Iterate through all detected cars
    rects = []
    rect_history = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = (((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
        print(bbox)
        rects.append(bbox)
        rect_history.append([np.min(nonzerox), np.min(nonzeroy),np.max(nonzerox), np.max(nonzeroy)])
        
    # Return final rectangles
    return  rects, rect_history

#This method was duplicated from lesson
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Define a class to store data from video
class Vehicles():
    def __init__(self):
        # history of rectangles previous n frames
        self.search_rects = []
        #collects best rectangles for several frames
        self.hist_rects = []
        #collects rectangles for several frames
        self.all_rects = [] 
        
    def add_frame_rects(self, rects):
        self.all_rects.append(rects)
        if len(self.all_rects) > 7:
            # throw out oldest rectangle set(s)
            self.all_rects = self.all_rects[len(self.all_rects)- 7:]

    def add_rects(self, rects):
        self.hist_rects.append(rects)
        if len(self.hist_rects) > 7:
            # throw out oldest rectangle set(s)
            self.hist_rects = self.hist_rects[len(self.hist_rects)- 7:]

    def add_superrects(self, rects, n=1):
        self.search_rects.append(rects)
        #print(self.search_rects)
        if len(self.search_rects) > n:
            # throw out oldest rectangle set(s)
            self.search_rects = self.search_rects[len(self.search_rects)- n:]

def process_image(img):
    #searching for vehicles in main region
    rectangles = []
    xstart = 0
    xstop = 1200
    
    #ystart = 396 #height=96
    #ystop = 492
    #scale = 1.0
    #rectangles.append(find_cars(img, ystart, ystop, xstart, xstop, scale, colorspace, svc, scaler, orient, pix_per_cell, cell_per_block, None, None, hog_channel))
    
    ystart = 396 #height=128
    ystop = 556 #524
    scale = 1.3 #1.2
    rectangles.append(find_cars(img, ystart, ystop, xstart, xstop, scale, colorspace, svc, scaler, orient, pix_per_cell, cell_per_block, None, None, hog_channel))
    
    ystart = 396 #height = 160
    ystop = 606 #556
    scale = 2 #2.0
    rectangles.append(find_cars(img, ystart, ystop, xstart, xstop, scale, colorspace, svc, scaler, orient, pix_per_cell, cell_per_block, None, None, hog_channel))
    
    rects = [item for sublist in cars.hist_rects for item in sublist]
    cars_rects = [item for sublist in cars.search_rects for item in sublist]

    if len(cars_rects) > 0:
        for bbox in cars_rects:
            print(bbox)
            if  bbox[2] - bbox[0] > 32:
                scale = 0.5
                rectangles.append(find_cars(img, bbox[1], bbox[3], bbox[0], bbox[2], scale, colorspace, svc, scaler, orient, pix_per_cell, cell_per_block, None, None, hog_channel))
    
    #flatten a list of lists. All rectangles found in a frame.
    rectangles = [item for sublist in rectangles for item in sublist] 
    #print(rectangles)
    #visual_rect = [item for sublist in cars.hist_rects for item in sublist]

    # Test out the heatmap
    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, rectangles)
    #rects = [item for sublist in cars.hist_rects for item in sublist]
    #print('rects',rects)
    heatmap_img = add_heat(heatmap_img, rects[0:3])

    heatmap_img = apply_threshold(heatmap_img, 3+len(rects)//6) #3+len(rects)//9)
    ratio = 250/heatmap_img.max()#17
    h_img = cv2.applyColorMap((heatmap_img*ratio).astype(np.uint8), colormap=cv2.COLORMAP_HOT)
    h_img = cv2.cvtColor(h_img, cv2.COLOR_BGR2RGB)
    h_img = cv2.resize(h_img, (640, 360))
    #cv2.imwrite('heatmap2/heatmap_labeled'+str(time.time())+'.jpg', heatmap_img)

    #print('threshold', len(rects)) #11    
    labels = label(heatmap_img)
    
    #print(labels[1], 'cars found')
    
    #calculate rectangles by labels in the current frame
    visual_rect, hist_rects = calculate_labeled_boxes(img, labels)
    
    #print(cars.hist_rects)
    
    if len(visual_rect) > 0:
        cars.add_rects(visual_rect)
        cars.add_frame_rects(hist_rects)
    hist_rects = [item for sublist in cars.all_rects for item in sublist]

    print(labels[1],' real cars found')
    draw_img = draw_labeled_bboxes(img, visual_rect)
    recent_boxes = np.array(hist_rects).tolist()
    if len(hist_rects) > 0:
        print(recent_boxes)
        boxes = cv2.groupRectangles(recent_boxes, 3, .1)
        print(boxes)
        # draw rectangles if found
        if len(boxes[0]) != 0:
            cars.add_superrects(boxes[0])
            for box in boxes[0]:
                cv2.rectangle(draw_img, (box[0], box[1]), (box[2],box[3]), (0,0,255), 2)

    #if len(visual_rect) > 0:
    #    
    
    # Return the image
    draw_img[0:360, 0:640, :] = h_img
    return draw_img

cars = Vehicles()

test_out_file2 = 'project_video_out_2.mp4'
clip_test2 = VideoFileClip('project_video.mp4')
clip_test_out2 = clip_test2.fl_image(process_image) #.subclip(4, 50)
clip_test_out2.write_videofile(test_out_file2, audio=False)
