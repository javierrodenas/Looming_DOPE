#!/usr/bin/env python

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file starts a ROS node to run DOPE, 
listening to an image topic and publishing poses.
"""

from __future__ import print_function

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw

import sys 
sys.path.append("inference")
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector
import simplejson as json
#from mmdet.apis import init_detector, inference_detector
#import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
import random
import cv2



def getModelInference(img_path):

    # Specify the path to model config and checkpoint file
    config_file = 'media/HDD_4TB/javi/FoodChallenge/food-recognition-challenge-starter-kit/mmdetection/looming/looming.py'
    checkpoint_file = 'media/HDD_4TB/javi/FoodChallenge/food-recognition-challenge-starter-kit/mmdetection/looming/latest.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    result = inference_detector(model, img_path)

    return result
    # visualize the results in a new window
    #result_image = model.show_result(img, result, score_thr=0.5, bbox_color=(0, 0, 0))

    # or save the visualization results to image files
    #model.show_result(img, result, out_file='result1.jpg')


def readImg(img_path):

    img = mmcv.imread(img_path)
    img = img.copy()

    return img


def chooseColor(available_colors, colors_already_used):

    color_to_use = None
    for index_color, color in enumerate(available_colors):
        if index_color not in colors_already_used:
            colors_already_used.append(index_color)
            color_to_use = color


    if color_to_use is None:
        colors_already_used = [0]
        color_to_use = available_colors[0]

    return color_to_use, colors_already_used


def processImage(img, result):

    blue = (255, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 233, 255)
    available_colors = [blue, red, green, yellow]

    # labels
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    scores = bboxes[:, -1]
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)



    #segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    #img[img>=0] = 0
    colors_already_used = []
    for index, (label, segmentation, score, bbox) in enumerate(zip(labels, segms, scores, bboxes)):
        print("LABEl", label)
        print("SCORE", score)

        if score > 0.6:
            color_to_use, colors_already_used = chooseColor(available_colors, colors_already_used)

            if label == 0: # Lateral
                #img[(segmentation==True)] = (0, 233, 255) #amarillo
                img[(segmentation == True)] = blue

            else: # Main
                #img[(segmentation == True)] = (192,192,192)
                pass


    for bbox in bboxes:
        if bbox[-1] > 0.6:
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, (0,0,0), thickness=1)

            center_x = (bbox[0] + (bbox[2] - bbox[0]) / 2)
            center_y = (bbox[1] + (bbox[3] - bbox[1]) / 2)
            img = cv2.circle(img, (int(center_x), int(center_y)), radius=2, color=red, thickness=20)


    mmcv.imwrite(img, '/home/javi/Desktop/inference/test_color.png')


    return img

def runMainLoop():

    img_path = 'color/color34.png'
    result = getModelInference(img_path)
    img = readImg(img_path)
    img_result = processImage(img, result)

    return img_result

class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius
            ]
            self.draw.ellipse(xy,
                              fill=point_color,
                              outline=point_color
                              )

    def draw_cube(self, points, color=(255, 0, 0)):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """

        # draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)


class DopeNode(object):
    """ROS node that listens to image topic, runs DOPE, and publishes DOPE results"""
    def __init__(self,
            config, # config yaml loaded eg dict
        ):
        self.pubs = {}
        self.models = {}
        self.pnp_solvers = {}
        self.pub_dimension = {}
        self.draw_colors = {}
        self.dimensions = {}
        self.class_ids = {}
        self.model_transforms = {}
        self.meshes = {}
        self.mesh_scales = {}

        self.input_is_rectified = config['input_is_rectified']
        self.downscale_height = config['downscale_height']

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config['thresh_angle']
        self.config_detect.thresh_map = config['thresh_map']
        self.config_detect.sigma = config['sigma']
        self.config_detect.thresh_points = config["thresh_points"]

        # For each object to detect, load network model, create PNP solver, and start ROS publishers
        print(config['weights'])
        for model in config['weights']:
            print(model)
            self.models[model] = \
                ModelData(
                    model,
                    config['weights'][model],
                    architecture = config['architectures'][model]
                )
            self.models[model].load_net_model()
            print('loaded')

            try:
                self.draw_colors[model] = tuple(config["draw_colors"][model])
            except:
                self.draw_colors[model] = (0,255,0)
            self.dimensions[model] = tuple(config["dimensions"][model])
            self.class_ids[model] = config["class_ids"][model]

            self.pnp_solvers[model] = \
                CuboidPNPSolver(
                    model,
                    cuboid3d=Cuboid3d(config['dimensions'][model])
                )


        # print("Running DOPE...  (Listening to camera topic: '{}')".format(config['~topic_camera')))
        print("Ctrl-C to stop")

    def image_callback(self, 
        img, 
        camera_info, 
        img_name = "00000.png", # this is the name of the img file to save, it needs the .png at the end
        output_folder = 'out_inference', # folder where to put the output
        ):
        img_name = str(img_name).zfill(5)
        """Image callback"""

        # img = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")

        # cv2.imwrite('img.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # for debugging

        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            P = np.matrix(camera_info['projection_matrix']['data'], dtype='float64').copy()
            P.resize((3, 4))
            camera_matrix = P[:, :3]
            dist_coeffs = np.zeros((4, 1))
        else:
            # TODO
            camera_matrix = np.matrix(camera_info.K, dtype='float64')
            camera_matrix.resize((3, 3))
            dist_coeffs = np.matrix(camera_info.D, dtype='float64')
            dist_coeffs.resize((len(camera_info.D), 1))

        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)))

        for m in self.models:
            self.pnp_solvers[m].set_camera_intrinsic_matrix(camera_matrix)
            self.pnp_solvers[m].set_dist_coeffs(dist_coeffs)

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)


        # dictionary for the final output
        dict_out = {"camera_data":{},"objects":[]}

        for m in self.models:
            # Detect object
            results, beliefs = ObjectDetector.detect_object_in_image(
                self.models[m].net,
                self.pnp_solvers[m],
                img,
                self.config_detect
            )

            # continue
            # Publish pose and overlay cube on image
            for i_r, result in enumerate(results):
                if result["location"] is None:
                    continue
                print(result)
                loc = result["location"]
                ori = result["quaternion"]
                
                print(loc)

                dict_out['objects'].append({
                    'class':m,
                    'location':np.array(loc).tolist(),
                    'quaternion_xyzw':np.array(ori).tolist(),
                    'projected_cuboid':np.array(result['projected_points']).tolist(),
                })
                # print( dict_out )

                # transform orientation
                # TODO 
                # transformed_ori = tf.transformations.quaternion_multiply(ori, self.model_transforms[m])

                # rotate bbox dimensions if necessary
                # (this only works properly if model_transform is in 90 degree angles)
                # dims = rotate_vector(vector=self.dimensions[m], quaternion=self.model_transforms[m])
                # dims = np.absolute(dims)
                # dims = tuple(dims)

                # Draw the cube
                if None not in result['projected_points']:
                    points2d = []
                    for pair in result['projected_points']:
                        points2d.append(tuple(pair))
                    draw.draw_cube(points2d, self.draw_colors[m])
        # save the output of the image. 
        im.save(f"{output_folder}/{img_name}.png")

        # save the json files 
        with open(f"{output_folder}/{img_name.replace('png','json')}", 'w') as fp:
            json.dump(dict_out, fp)

            

def rotate_vector(vector, quaternion):
    q_conj = tf.transformations.quaternion_conjugate(quaternion)
    vector = np.array(vector, dtype='float64')
    vector = np.append(vector, [0.0])
    vector = tf.transformations.quaternion_multiply(q_conj, vector)
    vector = tf.transformations.quaternion_multiply(vector, quaternion)
    return vector[:3]

if __name__ == "__main__":

    import argparse
    import yaml 
    import glob 
    import os 

    parser = argparse.ArgumentParser()
    parser.add_argument("--pause",
        default=10,
        help='pause between images')
    parser.add_argument("--showbelief",
        action="store_true",
        help='show the belief maps')
    parser.add_argument("--dontshow",
        action="store_true",
        help='headless mode')
    parser.add_argument("--outf",
        default="out_experiment",
        help='where to store the output')
    parser.add_argument("--data",
        default=None,
        help='folder for data images to load, *.png, *.jpeg, *jpg')
    parser.add_argument("--config",
        default="config_inference/config_pose.yaml",
        help='folder for the inference configs')
    parser.add_argument("--camera",
        default="config_inference/camera_info.yaml",
        help='camera info file')
    parser.add_argument('--realsense',
        action='store_true',
        help='use the realsense camera')



    #runMainLoop()

    opt = parser.parse_args()

    # load the configs
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)
    
    # setup the realsense
    if opt.realsense:
        import pyrealsense2 as rs
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)


    # create the output folder
    print (f"output is located in {opt.outf}")
    try:
        shutil.rmtree(f"{opt.outf}")
    except:
        pass

    try:
        os.makedirs(f"{opt.outf}")
    except OSError:
        pass


    # load the images if there are some
    imgs = []
    imgsname = []

    if not opt.data is None:
        videopath = opt.data
        for j in sorted(glob.glob(videopath+"/*.png")):
            imgs.append(j)
            imgsname.append(j.replace(videopath,"").replace("/",""))
    else:
        if not opt.realsense:
            cap = cv2.VideoCapture(0)

    # An object to run dope node
    dope_node = DopeNode(config)


    # starting the loop here
    i_image = -1

    """

    if opt.data is not None:
        for image_id in range(0, len(imgsname)):
            
            
    """


    while True:
        i_image+=1
        
        # Capture frame-by-frame

        if not opt.data:
            if opt.realsense:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
            else:
                key = cv2.waitKey(1)
                if key & 0xFF == ord('c') or key == 27:
                    ret, frame = cap.read()

            img_name = i_image
        else:
            if i_image >= len(imgs):
                i_image =0
            print(imgs)
            frame = cv2.imread(imgs[i_image])
            print(f"frame {imgsname[i_image]}")
            img_name = imgsname[i_image]

        frame = frame[...,::-1].copy()
        
        # call the inference node
        dope_node.image_callback(
            frame, 
            camera_info,
            img_name = img_name,
            output_folder = opt.outf)

        break
        """
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        """