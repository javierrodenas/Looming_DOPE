# Looming

<img src="https://github.com/javierrodenas/Looming_DOPE/blob/main/DOPE/github_img/Looming.PNG" width="50%" height="50%">

### Description

Combining these two algorithms, a solution capable of solving the production requirements has been implemented. First of all, isolating the pieces from the rest of the environment using SCNet means eliminating unwanted noise or light changes before detecting the location and orientation of the piece.


### Requirements

Python 3.X version


| Library | Version  |
| :---:   | :-: |
| mmdet | 2.12.0 |
| mmcv-full | 1.3.3 |
| pyrr | 0.10.3 |
| simplejson | 3.17.6 |


### Configuration
The combination of the two models is in the /SCNET/ directory with the following structure:

```
│Looming_DOPE/
├──DOPE/
├──SCNET/
│  ├── inference/
│  │   ├── config_inference/
│  │   │   ├── camera_info.yaml
│  │   │   ├── config_pose.yaml
│  ├── out_experiment/
│  ├── inference.py
│  ├── models.py
│  ├── requirements.txt
```

First of all, the path should be configured in config_pose.yaml where we will define the checkpoints to be used by the DOPE, for example:

```
“looming_piece": "/Documents/Looming_DOPE/DOPE/weights/mark_120.pth"
```
In our case, we have different DOPE models to be able to validate the prediction with multiple models. (looming_piece, looming_piece1, looming_piece2…)


On the other hand, SCNet checkpoints for pieces segmentation should be set up on looming_inference.py using the following code lines (already prepared in the .py file)
```
config_file = '/Documents/Looming_DOPE/SCNET/looming/looming.py'
checkpoint_file = '/Documents/Looming_DOPE/SCNET/looming/latest.pth'
```


### Usage
Once the checkpoints have been set up, we launch the code using line:

Inference using a folder with RGB images:
```
python3 looming_inference.py --data <path>
```
Inference using realsense:
```
python3 looming_inference.py  --realsense
```


<img src="https://github.com/javierrodenas/Looming_DOPE/blob/main/DOPE/github_img/complete_loop.png">

## Acknowledgements

Thanks to all the institutions that collaborate in this project and are making it possible to meet the objectives

## Reference

__[mmdetection](https://mmdetection.readthedocs.io/en/latest/api.html) - MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project__
__[DOPE](https://github.com/NVlabs/Deep_Object_Pose) - Deep Object Pose Estimation (DOPE) – ROS inference (CoRL 2018)__
