**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1-1]: ./output_images/image0000.png
[image1-2]: ./output_images/image8.png
[image2-1]: ./output_images/image0000_hog.jpg
[image2-2]: ./output_images/image7_hog.jpg
[image3-1]: ./output_images/figure_1.png
[image3-2]: ./output_images/figure_2.png
[image4]: ./output_images/examples.jpg
[image5]: ./output_images/examples2.jpg
[image6]: ./output_images/example3.jpg
[image7]: ./output_images/example4.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 59 through 135 `extract_features()` in `lesson_functins.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1-1] ![alt text][image1-2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Vehicle image example  
![alt text][image2-1]

Non-vehicle image example  
![alt text][image2-2]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters such as orientations, pixcels_per_cell and cells_per_block.
In the end, I just found the combination above gave the training accuracy 99.7 %.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features of three channels, color features of three channels and spatial features. For spatial features, I used 16 x 16 pixel binning dimensions.
It is in the code line 32 through 73 `train_classifier()` in `main_process.py`

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the image area whose Y coordinate is between 400 px and 780 px by checking the positions of the road manually by an image processing tool. The cell per step of sliding window is two. The cell per block is two. I used two size of sliding window. One is 64 pix x 64 pix. The other one is 96 pix x 96 pix. I decided these parameters by experimenting the pipeline many times with test images.

64 pix x 64 pix (One cell is 8 pix. Sliding 2 cells(16 pix))
![alt text][image3-1]

96 pix x 96 pix (One cell is 12 pix. Sliding 2 cells (24 pix))
![alt text][image3-2]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)  
Here's a [link to my video result](./CarDetect_Lane.mp4)
I combined the car detection with the advanced lane detection from the project #4.


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a frame of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes overlaid on the output of `scipy.ndimage.measurements.label()`:

### Here is a frame and its corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from the frame above:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Since I did not track the heat map position between frames, the detection windows on the image was a little bit shaky. By implementing the heat map tracking, the detection windows could have moved more smoothly.
