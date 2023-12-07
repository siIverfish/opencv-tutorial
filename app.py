import cv2 as opencv
import numpy

from contour_lib import *

image = opencv.imread("test_8.jpg")
blurred_image = opencv.medianBlur(image, ksize=11)
hsv_blurred_image = opencv.cvtColor(image, opencv.COLOR_BGR2HSV)
binary_image = opencv.inRange(
    hsv_blurred_image, 
    lowerb=numpy.array([18, 0, 0]), 
    upperb=numpy.array([20, 250, 264])
)



contours, _ = opencv.findContours(
    binary_image, 
    opencv.RETR_TREE, 
    opencv.CHAIN_APPROX_SIMPLE
)

largest_object = get_maximum_contour(contours)
center = get_contour_center(largest_object)

image_with_target_dot = opencv.circle(
    image, 
    center, 
    radius=5, 
    color=(0, 0, 255), 
    thickness=-1
)

opencv.imshow("Behold unto ye, the CONE (hopefully!!)", image_with_target_dot)
opencv.waitKey(0)
opencv.destroyAllWindows()