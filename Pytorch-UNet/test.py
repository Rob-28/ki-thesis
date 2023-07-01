import argparse
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from utils.data_loading import BasicDataset, CarvanaDataset
from unet import UNet


import matplotlib.pyplot as plt

import predict

def plot_img_ref_and_mask(img, ref, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1, figsize=(20,5))
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Input reference')
    ax[1].imshow(ref)
    for i in range(1, classes):
        ax[i + 1].set_title(f'Prediction')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def get_validation_database(dir_img, dir_mask, img_scale, val_percent = 0.1, seed = 0):
    """
    Create the training-validation split in the exact same manner as during training.
    """
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    return(dataset)

if __name__ == '__main__':
    args = predict.get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    validate_set = len(args.input) == 1 and str(args.input[0]).endswith('/')

    if not validate_set:
        in_files = args.input
        in_ref = [x.replace("imgs", "masks") for x in args.input]
        out_files = predict.get_output_filenames(args)

    net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu' # force evaluation on CPU (useful if GPU is working on other task)

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    IoUs = []
    names = []

    if not validate_set:
        for i, (image_path, mask) in enumerate(zip(in_files, in_ref)):
            logging.info(f'Predicting image {image_path} ...')
            img = Image.open(image_path)
            reference = Image.open(mask).convert("L")
            reference_array = np.array(reference)

            mask = predict.predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)
            
            overlap = (reference_array==1) * (mask==1) # Logical AND
            union = (reference_array==1) + (mask==1) # Logical OR

            IoU = overlap.sum()/float(union.sum())
            print(IoU)
            IoUs.append(IoU)
            names.append(image_path)

            if not args.no_save:
                out_filename = out_files[i]
                mask = predict.mask_to_image(mask, mask_values)
                mask.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if args.viz:
                logging.info(f'Visualizing results for image {image_path}, close to continue...')
                plot_img_ref_and_mask(img, reference, mask)
    
    if validate_set:
        mask_dir = args.input[0] + "../masks/"

        val_set = get_validation_database(args.input[0], mask_dir, args.scale)
        
        for i, pair in enumerate(val_set):
            logging.info(f'Predicting image {i} ...')

            image_array = (255*pair["image"]).numpy().astype(np.uint8).transpose(1, 2, 0)

            if args.channels == 1:
                img = Image.fromarray(image_array[:,:,0])
            if args.channels == 2:
                img = Image.fromarray(image_array[:,:,:])
            if args.channels == 3:
                img = Image.fromarray(image_array[:,:,[2,1,0]])
            if args.channels == 4:
                img = Image.fromarray(image_array[:,:,[2,1,0,3]])
            

            reference_array = (pair["mask"]).numpy().astype(np.uint8)
            reference = Image.fromarray(reference_array)

            mask = predict.predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)
            
            overlap = (reference_array==1) * (mask==1) # Logical AND
            union = (reference_array==1) + (mask==1) # Logical OR

            print("Overlap:", overlap.sum())
            print("Union:", union.sum())

            if union.sum() > 0:
                IoU = overlap.sum()/float(union.sum())
            else:
                IoU = 1

            print(IoU)
            if not math.isnan(IoU):
                IoUs.append(IoU)
                names.append(pair["name"])

            if not args.no_save:
                # out_filename = out_files[i]
                # result = predict.mask_to_image(mask, mask_values)
                # result.save(out_filename)
                try:
                    os.mkdir("./output"+str(args.channels))
                except FileExistsError:
                    pass
                out_filename = os.path.join("./output"+str(args.channels)+"/", pair["name"].replace(".npy", ".png"))
                Image.fromarray((mask*255).astype('uint8')).convert("RGB").save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if args.viz:
                logging.info(f'Visualizing results for image {i}, close to continue...')
                plot_img_ref_and_mask(img, reference, mask)

    print("Average IoU:", np.mean(IoUs))
    print("SD IoU:", np.std(IoUs))

    names = np.expand_dims(np.array(names), 1)
    IoUs = np.expand_dims(np.array(IoUs), 1)
    np.savetxt("../csv/evaluation-unet-"+str(args.channels)+"channels.csv", np.hstack((names, IoUs)), fmt='%s', delimiter=',', header="name,IoU", comments='')
