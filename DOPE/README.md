# Looming_DOPE

<img src="https://github.com/javierrodenas/Looming_DOPE/blob/main/github_img/Looming.PNG" width="50%" height="50%">


### Description

DOPE performs the detection and estimation of the 3D objects position from a single RGB image. Use a deep learning approach to predict the key points of the images focusing on the corners of the object and the center of the 3D delimiting box. After that, it applies a PnP post processing to estimate the 3D position.


### Dataset generator

<img src="https://github.com/javierrodenas/Looming_DOPE/DOPE/blob/main/github_img/generator.PNG" width="80%" height="80%">

### Dataset generation

<img src="https://github.com/javierrodenas/Looming_DOPE/DOPE/blob/main/github_img/dataset_generation.PNG" width="80%" height="80%">


### Image generated

<img src="https://github.com/javierrodenas/Looming_DOPE/DOPE/blob/main/github_img/looming_piece.png" width="80%" height="80%">


### Running

__1. Start ROS master__

```
cd ~/catkin_ws
source devel/setup.bash
roscore
```

__2. Start camera node (or start your own camera node)__
    
Realsense D435 & usb_cam node (./dope/config/config_pose.yaml):    

``` 
topic_camera: "/camera/color/image_raw"            #"/usb_cam/image_raw"
topic_camera_info: "/camera/color/camera_info"     #"/usb_cam/camera_info"
``` 

Start camera node:
   
``` 
roslaunch realsense2_camera rs_rgbd.launch  # Publishes RGB images to `/camera/color/image_raw`
```

__3. Start DOPE node__
    
```
roslaunch dope dope.launch [config:=/path/to/my_config.yaml]  # Config file is optional; default is `config_pose.yaml`
```

__4. Start rviz node__
```
rosrun rviz rviz
```


## Acknowledgements

Thanks to all the institutions that collaborate in this project and are making it possible to meet the objectives

## Reference

__[DOPE](https://github.com/NVlabs/Deep_Object_Pose) - Deep Object Pose Estimation (DOPE) – ROS inference (CoRL 2018)__
