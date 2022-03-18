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

```
cd tools
python3 tools/train.py looming.py  
```

## Acknowledgements

Thanks to all the institutions that collaborate in this project and are making it possible to meet the objectives

## Reference

__[mmdetection](https://mmdetection.readthedocs.io/en/latest/api.html) - MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project__
__[DOPE](https://github.com/NVlabs/Deep_Object_Pose) - Deep Object Pose Estimation (DOPE) – ROS inference (CoRL 2018)__
