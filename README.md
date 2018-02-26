**Vehicle Detection Project**

The goals/steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG.JPG
[image2]: ./output_images/test1.jpg
[image3]: ./output_images/test2.jpg
[image4]: ./output_images/test3.jpg
[image5]: ./output_images/test4.jpg
[image6]: ./output_images/test5.jpg
[image7]: ./output_images/test6.jpg
[image8]: ./output_images/test_image_HLS_1.jpg
[image9]: ./output_images/test_image_HLS_3.jpg
[image10]: ./output_images/test_image_r1.0.jpg
[image11]: ./output_images/test_image_all_rectangles_1.jpg
[image12]: ./output_images/heatmap_labeled1.jpg
[image13]: ./output_images/heatmap_labeled2.jpg
[image14]: ./output_images/heatmap_labeled3.jpg
[image15]: ./output_images/heatmap_labeled4.jpg
[image16]: ./output_images/heatmap_labeled5.jpg
[image17]: ./output_images/heatmap_labeled6.jpg
[image18]: ./output_images/result.JPG
[video1]: ./project_video_out_2_last.mp4

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the IPython notebook(P5.ipynb) included in the archive.  

I started by reading in all the `vehicle` and `non-vehicle` images. The numbers of 'vehicle' and 'non-vehicle' images are quite similar and dataset in general looks balanced. I plot several images from each group in the notebook to analyze the difference between classes. 
In cell 5 of the notebook, I extracted hog features for random 'vehicle' and 'non-vehicle' to show the difference between classes. Here is an example of one image of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

####2. Explain how you settled on your final choice of HOG parameters.
I then explored different color spaces and different hog extraction parameters on test images. I tried various combinations of parameters and found that, first of all, 3 channels shall be used for feature extraction. In a case when only one channel is used classification results become worse. Below is an example of vehicle detection in HLS color space with 1 channel and 3 channels used for feature extraction. It is possible to see that black car is not detected when only one channel is used.
![alt text][image8]
![alt text][image9] 
Second, сhanging an orientation between 8 and 11 didn't show significant influence on detection results but has an influence on the set of features and calculation speed. 
Third, the preferable color space is HLS. 
Forth, I spent a lot of time finding appropriate dataset size for required classifier performance. The Linear SVM classifier didn't give me required performance but classifier with RBF kernel gave me low calculation speed when the size of a feature vector was around several thousand features. 
My previous version was with the cell_per_block = 1 because it significantly reduces the number of features and speeds up calculations but this approach doesn't take into account effects of luminance and others.  
The final set of the parameters is the following:
cspace = 'HLS'
orient = 9
pix_per_cell = 16 
cell_per_block = 2
hog_channel = 'ALL' 


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I experienced calculation troubles with dataset preparation. Because of that I implemented functions for dataset preparation separately in file prepare_data.py(lines:142-171) included in the archive. I normalized feature vector according to Tips and Tricks recommended in lectures using Standard Scaler. 
Training of the classifier is presented in other file included in the archive - video.py (lines: 93-105)
I trained a linear SVM using sklearn LinearSVC classifier. The best performance I received was 
HLS(orient=9_pixels=8_cells=3), C=10 - 0.9687. But it was not enough for the current task. RBF kernel gave me best performance but it was slow. I tuned C using recommendations given in 1. 
After that, I decided to use the approach suggested in [2] and created the dataset with HLS(orient=9_pixels=16_cells=2). 
Linear classifier in this case also gave while C=0.01 - 0.9757. But RBF provided me the best performance I have seen before:kernel=rbf, C=10, gamma = auto - 0.9928

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search was implemented based on the function find_cars(video.py, lines: 125-211) that was suggested in lectures, but I little bit changed it. I added parameters xstart and xstop for small rectangles to search vehicles. 
I analyzed test images and video and selected the main region for searching: ystart = 396, ystop = 656.
For test image searching I selected several scales (1,1.3,2,2.9) and defined several regions for it. But for the video I decided to search for big scale and then search squares with detected cars in details with small size. And I also changed parameter cells_per_step = 1. The current video scales are 1.3 and 2.0. 

I search the main region with a big scale. When I find something interesting I search for details. I selected 75% of window overlapping because it gives better search results. I also added image scaling as it suggested in the Tips and tricks. 

The example of the image with scale = 1 painted is presented below:   

![alt text][image10]

A number of rectangles of different sizes were painted in the next figure to display sliding window search. As it is possible to see, vehicles are detected by rectangles of different sizes: 
![alt text][image11]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

At the end I searched on 4 scales using HLS 3-channel HOG features in the feature vector, which provided a nice result. Here are some example images:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

It is possible to see that there is a problem with small white car in the third image. In the video, I experienced the same problem and then augmented dataset with several hundreds white cars. 
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out_2_last.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The following pipeline was implemented (video.py, lines: 323, 416):
1. I take a frame and search for the vehicles in several big regions with big scales(1.3, 2.0). 
2. I record the positions of positive detections in the frame of the video.
3. From the positive detections, I created a heat map and then thresholded that map to identify vehicle positions in the frame. The positive detections from current frame are augmented by raw data from previous several frames. And also I perform an additional search for trusted rectangles that are filtered after applying history. 
4. I then used `scipy.label()` to identify individual blobs in the heat map.  I then assumed each blob corresponded to a vehicle.  I constructed rectangles to cover the area of each blob detected.  
5. After that, I augment heatmap and apply there only rectangles found on the previous step and rectangles from history. After that I thresholded them, constructed and painted rectangles to cover the area of each blob detected. Using this way I receive trusted rectangles and avoid unstable rectangles.
Here's an example result showing the heat map from a series of frames of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image18]

---

###Discussion

####1. Briefly discuss any problems/issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had problems with white car detection. For the detection purposes, I created a set of images and augmented initial dataset. 
I had calculation problems with dataset preparation and later with video processing. To increase calculation speed I decreased dataset size using parameters. 
For frame processing, I receive the better speed - about one frame per second. It is to slow for the realtime. But maybe improved by stronger hardware. 
Using current approach I don't receive false positives, but sometimes filter has "inertion": it shows the region where the car was. This problem may be solved by reducing a number of history frames that I add to the heat map. In this case, my filter becomes more sensitive. The other approach is to estimate car direction and search for the forecast region. 
Currently, I search for details in a rectangle that was defined from the previous frame. But this way may be improved by extending of the search region with estimation future car position. But I am sure that it will reduce calculation speed.
Currently, my approach is the compromise between calculation speed and filter sensibility. 
Future improvements may be the following:
Add vehicle tracking using Kalman filter. 
Forecast region where to search in details.
Improve quality is using CNN, for example, YOLO, SSD or something other. 
Helpful links: 
[1]https://neerajkumar.org/writings/svm/
[2]https://medium.com/@matlihan/a-trick-to-quickly-explore-hog-features-for-udacity-vehichle-detection-and-tracking-project-0-991-8b6f682a0b01
