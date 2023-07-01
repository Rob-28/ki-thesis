# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import glob
import math
import os
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
from matplotlib import pyplot
import torch
import argparse
import whisper
import numpy as np

from gradio import processing_utils
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

from tasks import *

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/seem_focall_lang.yaml", metavar="FILE", help='path to config file', )
    args = parser.parse_args()

    return args

'''
build args
'''
args = parse_option()
opt = load_opt_from_config_files(args.conf_files)
opt = init_distributed(opt)

# META DATA
cur_model = 'None'
if 'focalt' in args.conf_files:
    pretrained_pth = os.path.join("seem_focalt_v2.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v2.pt"))
    cur_model = 'Focal-T'
elif 'focal' in args.conf_files:
    pretrained_pth = os.path.join("seem_focall_v1.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
    cur_model = 'Focal-L'

'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)


@torch.no_grad()
def inference(image, task, *args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        if 'Video' in task:
            return interactive_infer_video(model, None, image, task, *args, **kwargs)
        else:
            return interactive_infer_image(model, None, image, task, *args, **kwargs)

if __name__ == "__main__":
    image_dir = "../../Pytorch-UNet/data/test/rgb/"
    mask_dir = "../../Pytorch-UNet/data/test/masks/"

    images = [file for file in os.listdir(image_dir)]
    images.sort()
    masks = [file for file in os.listdir(mask_dir)]
    masks.sort()

    IoUs = []
    names = []

    try:
        os.mkdir("./output")
    except FileExistsError:
        pass

    for image_path, mask_path in zip(images, masks):
        print(image_path)
        if image_path.endswith('.npy'):
            np_img = np.load(os.path.join(image_dir, image_path))
            img = Image.fromarray(np_img)
        # else:
        #     img = Image.open(os.path.join(image_dir, image_path))
        #     np_img = np.array(img)

        if image_path.endswith('.npy'):
            np_reference = np.load(os.path.join(mask_dir, mask_path)) > 0
            reference = Image.fromarray(np_reference)
        # else:
        #     reference = Image.open(os.path.join(mask_dir, mask_path))
        #     np_reference = np.array(reference) / 255

        image = {
            "image": img,
            "mask": reference
        }


        demo = False

        if demo:
            (result, _) = inference(image, [])

            pyplot.imshow(result)
            pyplot.title(image_path)
            pyplot.show()
        else:
            (_, result) = inference(image, ["Text"], reftxt="water")
            result = np.squeeze(result)

            overlap = (np_reference) * (result==1) # Logical AND
            union = (np_reference) + (result==1) # Logical OR

            if union.sum() > 0:
                IoU = overlap.sum()/float(union.sum())
            else:
                IoU = 1

            print(IoU)
            if not math.isnan(IoU):
                IoUs.append(IoU)
                names.append(image_path)
            
            Image.fromarray(result*255).convert("RGB").save(os.path.join("./output/", image_path.replace(".npy", ".png")))
            

    print("Average IoU:", np.mean(IoUs))
    print("SD IoU:", np.std(IoUs))

    IoUs = np.expand_dims(np.array(IoUs), 1).astype(str)
    names = np.expand_dims(np.array(names), 1).astype(str)
    np.savetxt("../../csv/evaluation-SEEM.csv", np.hstack((names, IoUs)), fmt='%s', delimiter=',', header="name,IoU", comments='')

