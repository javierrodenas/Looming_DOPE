import cv2
import glob, os, errno

# Replace mydir with the directory you want
mydir = r'/home/javi/Desktop/FoodChallenge/models/research/deeplab/datasets/pascal_voc_seg/SegmentationClass'
mydir_color = '/home/javi/Desktop/FoodChallenge/models/research/deeplab/datasets/pascal_voc_seg/SegmentationClassColor/'


for fil in glob.glob("/home/javi/Desktop/FoodChallenge/models/research/deeplab/datasets/pascal_voc_seg/SegmentationClass/*.png"):
    image = cv2.imread(fil,0) 
    gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # convert to greyscale
    cv2.imwrite(mydir_color + fil.split('/')[-1],gray_image)