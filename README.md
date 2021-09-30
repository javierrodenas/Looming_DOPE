# Looming_DOPE



### Dataset generator

<img src="https://github.com/javierrodenas/Looming_DOPE/blob/main/github_img/generator.PNG" width="80%" height="80%">

### Dataset generation

<img src="https://github.com/javierrodenas/Looming_DOPE/blob/main/github_img/dataset_generation.PNG" width="80%" height="80%">


### Image generated

<img src="https://github.com/javierrodenas/Looming_DOPE/blob/main/github_img/looming_piece.png" width="80%" height="80%">


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


