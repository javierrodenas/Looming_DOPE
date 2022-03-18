# Looming_SCNET

<img src="https://github.com/javierrodenas/Looming_DOPE/blob/main/DOPE/github_img/Looming.PNG" width="50%" height="50%">

### Description

SCNet performs the segmentation of pieces from a single RGB image. Use a deep learning approach to predict the pixels that belong to a piece, getting as an output the mask per object.

### Dataset generation

<img src="https://github.com/javierrodenas/Looming_DOPE/blob/main/DOPE/github_img/labelme.PNG" width="80%" height="80%">

Tool [Labelme](https://github.com/wkentaro/labelme) used to label the images.

### Training

```
cd tools
python3 tools/train.py looming.py  
```

## Acknowledgements

Thanks to all the institutions that collaborate in this project and are making it possible to meet the objectives

## Reference

__[mmdetection](https://mmdetection.readthedocs.io/en/latest/api.html) - MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project
