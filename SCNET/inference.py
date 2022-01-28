from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
import random


def getModelInference(img_path):

    # Specify the path to model config and checkpoint file
    config_file = 'looming/looming.py'
    checkpoint_file = 'looming/latest.pth'

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
    for index, (label, segmentation, score) in enumerate(zip(labels, segms, scores)):
        print("LABEl", label)
        print("SCORE", score)

        if score > 0.6:
            color_to_use, colors_already_used = chooseColor(available_colors, colors_already_used)
            if label == 0: # Lateral
                #img[(segmentation==True)] = (0, 233, 255) #amarillo
                img[(segmentation == True)] = red
            """
            else: # Main
                img[(segmentation == True)] = blue
            """
    mmcv.imwrite(img, 'test_color.png')


def runMainLoop():

    img_path = 'color/color39.png'
    result = getModelInference(img_path)
    img = readImg(img_path)
    processImage(img, result)

if __name__ == "__main__":
    runMainLoop()