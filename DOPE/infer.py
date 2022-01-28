import numpy as np

import yaml
import torch
from src.dope.inference.cuboid import *
from src.dope.inference.cuboid_pnp_solver import CuboidPNPSolver
from src.dope.inference.detector import *
from os import listdir
# import pyrealsense2 as rs

from PIL import Image
from PIL import ImageDraw
import cv2
from datetime import datetime

### Code to visualize the neural network output

def DrawLine(g_draw, point1, point2, lineColor, lineWidth):
    '''Draws line on image'''
    # global g_draw
    if not point1 is None and point2 is not None:
        g_draw.line([point1, point2], fill=lineColor, width=lineWidth)


def DrawDot(g_draw, point, pointColor, pointRadius):
    '''Draws dot (filled circle) on image'''
    # global g_draw
    if point is not None:
        xy = [
            point[0] - pointRadius,
            point[1] - pointRadius,
            point[0] + pointRadius,
            point[1] + pointRadius
        ]
        g_draw.ellipse(xy,
                       fill=pointColor,
                       outline=pointColor
                       )


def DrawCube(g_draw, points, color=(255, 0, 0)):
    '''
    Draws cube with a thick solid line across
    the front top edge and an X on the top face.
    '''
    color = (255, 215, 0)
    lineColor1 = (255, 215, 0)
    lineWidthForDrawing = 2

    # draw front
    DrawLine(g_draw, points[0], points[1], color, lineWidthForDrawing)
    DrawLine(g_draw, points[1], points[2], color, lineWidthForDrawing)
    DrawLine(g_draw, points[3], points[2], color, lineWidthForDrawing)
    DrawLine(g_draw, points[3], points[0], color, lineWidthForDrawing)

    # draw back
    DrawLine(g_draw, points[4], points[5], color, lineWidthForDrawing)
    DrawLine(g_draw, points[6], points[5], color, lineWidthForDrawing)
    DrawLine(g_draw, points[6], points[7], color, lineWidthForDrawing)
    DrawLine(g_draw, points[4], points[7], color, lineWidthForDrawing)

    # draw sides
    DrawLine(g_draw, points[0], points[4], color, lineWidthForDrawing)
    DrawLine(g_draw, points[7], points[3], color, lineWidthForDrawing)
    DrawLine(g_draw, points[5], points[1], color, lineWidthForDrawing)
    DrawLine(g_draw, points[2], points[6], color, lineWidthForDrawing)

    # draw dots
    DrawDot(g_draw, points[0], pointColor=color, pointRadius=4)
    DrawDot(g_draw, points[1], pointColor=color, pointRadius=4)

    # draw x on the top
    DrawLine(g_draw, points[0], points[5], color, lineWidthForDrawing)
    DrawLine(g_draw, points[1], points[4], color, lineWidthForDrawing)

class Inference:
    def __init__(self):
        self.models = {}
        self.pnp_solvers = {}
        self.draw_colors = {}
        config_detect = None

    def load_config(self, config_name):
        yaml_path = 'config/{}'.format(config_name)
        with open(yaml_path, 'r') as stream:
            try:
                print("Loading DOPE parameters from '{}'...".format(yaml_path))
                params = yaml.load(stream)
                print('    Parameters loaded.')
            except yaml.YAMLError as exc:
                print(exc)

            # pub_dimension = {}

            # Initialize parameters
            matrix_camera = np.zeros((3, 3))
            matrix_camera[0, 0] = 640
            matrix_camera[1, 1] = 640
            matrix_camera[0, 2] = 640
            matrix_camera[1, 2] = 360
            matrix_camera[2, 2] = 1
            dist_coeffs = np.zeros((4, 1))

            """
            if "dist_coeffs" in params["camera_settings"]:
                dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
            """
            self.config_detect = lambda: None
            self.config_detect.threshold = 0
            self.config_detect.thresh_angle = 100
            self.config_detect.thresh_map = 0.01
            self.config_detect.sigma = 0
            self.config_detect.thresh_points = 0.01


            # For each object to detect, load network model, create PNP solver, and start ROS publishers
            for model in params['weights']:
                full_model_path = params['weights'][model]
                relative_model_path = full_model_path.split("package://dope/", 1)[1]
                print("model", model, relative_model_path)
                self.models[model] = \
                    ModelData(
                        model,
                        relative_model_path
                    )
                self.models[model].load_net_model()

                self.draw_colors[model] = tuple(params["draw_colors"][model])

                self.pnp_solvers[model] = \
                    CuboidPNPSolver(
                        model,
                        matrix_camera,
                        Cuboid3d(params['dimensions'][model]),
                        dist_coeffs=dist_coeffs
                    )

    def dope_on_webcam(self):
        # RealSense Start
        # pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # profile = pipeline.start(config)
        # # Setting exposure
        # s = profile.get_device().query_sensors()[1]
        # s.set_option(rs.option.exposure, exposure_val)
        cap = cv2.VideoCapture(0)

        while True:
            # Reading image from camera
            ret, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.dope_on_img(img)

    def dope_on_img_path(self, img_path):
        img = cv2.imread(img_path)

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.dope_on_img(img, wait_forever=True)

    def dope_on_dataset_path(self, dataset_path, dataset_size):

        for i in np.arange(0, dataset_size):
            img_path = dataset_path + "/" + "{:06n}".format(i) + ".png"
            print("path:", img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.dope_on_img(img)
            time.sleep(0.1)

    def dope_on_img(self, img, wait_forever=False):

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        g_draw = ImageDraw.Draw(im)

        for m in self.models:
            # Detect object
            try:
                results, vertex2 = ObjectDetector.detect_object_in_image(
                    self.models[m].net,
                    self.pnp_solvers[m],
                    img,
                    self.config_detect
                )
                # Overlay cube on image
                for i_r, result in enumerate(results):
                    # print("result", result)
                    if result["location"] is None:
                        continue
                    loc = result["location"]
                    ori = result["quaternion"]

                    # Draw the cube
                    if None not in result['projected_points']:
                        points2d = []
                        for pair in result['projected_points']:
                            points2d.append(tuple(pair))

                        # print("points2d", points2d)
                        print("num points:", len(points2d))
                        DrawCube(g_draw, points2d, self.draw_colors[m])

                # try and print the belief maps
                if vertex2 is not None:
                    for j in range(vertex2.size()[0]):
                        belief = vertex2[j].clone()
                        belief -= float(torch.min(belief).data.cpu().numpy())
                        belief /= float(torch.max(belief).data.cpu().numpy())
                        belief = torch.clamp(belief, 0, 1)
                        belief = torch.cat([belief.unsqueeze(0), belief.unsqueeze(0), belief.unsqueeze(0)]).unsqueeze(0)
                        temp = Variable(belief.clone())
                        array_belief = temp.data.squeeze().cpu().numpy().transpose(1, 2, 0) * 255
                        cv2.imshow('belief_' + str(j), array_belief)
                        if j < 5:
                            cv2.moveWindow('belief_' + str(j), j * 400, 0)
                        else:
                            cv2.moveWindow('belief_' + str(j), (j-5) * 400, 200)


            except Exception as e:
                print("CRASHED")
                print("Error: " + str(e))
                print("saving image...")
                dateTimeObj = datetime.now()
                dateStr = dateTimeObj.strftime("%H:%M:%S_%d_%m_%Y")
                print(dateStr)
                Image.fromarray(img).save(dateStr + ".png")

        open_cv_image = np.array(im)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Open_cv_image', open_cv_image)
        cv2.imwrite('cube4.png', open_cv_image)
        if wait_forever:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)


if __name__ == '__main__':
    # Settings
    config_name = "config_pose.yaml"
    exposure_val = 166

    infer = Inference()
    infer.load_config(config_name)
    # infer.dope_on_webcam()
    # infer.dope_on_img_path("imgs/crashed/crashed_img_12:37:10_29_09_2020.png")
    infer.dope_on_img_path("/home/javi/Desktop/000002.png")
    # infer.dope_on_dataset_path("/home/sebastian/datasets/kalo1.5_1_object_100000_400x400", 100000)