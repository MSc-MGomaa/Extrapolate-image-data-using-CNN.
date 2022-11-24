# Extrapolate-image-data-using-CNN
Predict the pixels' values of a cropped area of a given photo using Convolutional Neural Network

## First Part
<p align="justify">
In this project you should create a function ex4(image_array, crop_size, crop_center) that creates two input arrays and one target array from one input image. For this, your function
should crop out (=set values to zero) a part of the image, which will then become the target. Since it could be valuable information for our network to know which part was cropped out and
should be restored, we will also prepare an additional input channel that includes information about which pixels are in or outside the cropped-out rectangle.

In detail, your function should take the following keyword arguments:
* image_array: A numpy array of shape (X, Y) and arbitrary datatype, which contains the image data.
* crop_size: A tuple containing 2 odd int values. These two values specify the lengths of the rectangle that should be cropped-out in pixels for the two spatial dimensions X and
Y in that order.
* crop_center: A tuple containing 2 int values. These two values are the position of the center of the to-be cropped-out rectangle in pixels for the two spatial dimensions X and Y
in that order.

<p align="justify">
Your function should return a tuple (image_array, crop_array, target_array), where the returned image_array is a modified version of the original image_array that the function
gets as argument.

<p align="justify">
image_array should be modified such that the pixels in the cropped-out rectangle are set to 0, while the rest of the pixels remains unchanged. You may edit the original image_array
in-place or create a copy.

<p align="justify">
crop_array should be a numpy array of same shape and datatype as image_array, containing value 0 for pixels located outside the cropped-out rectangle and 1 for pixels located in the
cropped-out rectangle.

<p align="justify">
target_array should be a 2D numpy array of the same datatype as image_array, containing the values of the original image_array in the cropped-out rectangle.

<p align="justify">
The to-be cropped-out rectangle is specified via the center of the rectangle crop_center and the size of the rectangle in pixels crop_size. Theoretically, we could rotate the rectangle or
choose other forms to crop out but we will not consider these cases here.

Your function should raise a ValueError exception if
* image_array is not a 2D numpy array (see hints on how to check if an object is a numpy array instance).
* crop_size or crop_center do not contain exactly 2 objects. (You do not need to check the datatype of the objects, you can assume them to be integers.)
* The values in crop_size are even numbers.
* The minimal distance between the to-be cropped-out rectangle and the border of image_array is less than 20 pixels.

<p align="center">
<img width="800" height="350" src="https://github.com/MSc-MGomaa/Extrapolate-image-data-using-CNN./blob/3c0d1db878d5176813d7a35e517950ddeb74897f/CNN.png">

## Second Part

Challenge specifications:
* Samples considered in the challenge are grayscale images where a part of the image was cropped out.
* Your model should predict (=restore) the cropped-out pixel values.
* The images collected in exercise 1 will be the training set, however, you are free to include more images of your choosing into your training set.

Scoring will be performed as follows:
* You will be provided with test set images, where a part of the image was cropped out.
* The images and the center and size of the cropped-out part will be provided as pickle file (format see below).
* The images will have a width and height of at least 70 and at most 100 pixels.
* The width and height of the cropped-out rectangle will be odd integers of at least 5 and at most 21 pixels.
* The border between the cropped-out rectangle and the borders of the image will be at least 20 pixels, as in the first part.
* Predictive performance will be measured by the mean squared error. The error per sample will be the mean squared error of target vs. predicted pixels. The final loss of your model will be the mean over the sample errors in the test set.
* You will need to predict the missing pixel values for each sample and submit the predicted values to the challenge server as pickle files (format see below). The loss of your model will be computed on the server.


## Test set format. 
The test set will be provided as pickle file containing a dictionary with entries "images", "crop_sizes", and "crop_centers".

* <p align="justify"> "images" will be a list of numpy arrays of shape (X, Y) and datatype numpy.uint8, where each numpy array represents one image and X and Y are the spatial dimensions.
The images were read from grayscale .jpg images into numpy arrays using PIL and resized using the interpolation method PIL.Image.LANCZOS.
* <p align="justify"> "crop_sizes" will be a list of tuples with 2 elements per tuple of datatype int, where each tuple represents the crop_size.
* <p align="justify"> "crop_centers" will be a list of tuples with 2 elements per tuple of datatype int, where each tuple represents the crop_center.
* <p align="justify"> As in the first part, crop_size and crop_center are tuples specifying the length and position for dimension X and Y in the respective numpy array.
* <p align="justify"> All elements in the lists (images, crop_size, and crop_center) have the same order. That is, element at index 0 in the list in "images" is the numpy array containing the pixel values for the first image, element at index 0 in the list in "crop_sizes" is the crop_siz for the first image, and element at index 0 in the list in "crop_centers" is the crop_center for the first image. Elements at index 1 in the lists would represent the data for the second image, elements at index 2 the third image, and so on.

## An example from the experiments

<p align="center">
<img width="700" height="400" src="https://github.com/MSc-MGomaa/Extrapolate-image-data-using-CNN./blob/3c0d1db878d5176813d7a35e517950ddeb74897f/temp2.PNG">





