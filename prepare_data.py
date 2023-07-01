import math
import os
import random
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.augmentations.geometric.transforms import Perspective, ShiftScaleRotate
from albumentations.augmentations.transforms import RandomRain
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip
import numpy as np
from skimage import feature
from matplotlib import pyplot

split = {
    "train": 0.7,
    "validate": 0.15,
    "test": 0.15
}

assert (split["train"] + split["validate"] + split["test"]) == 1

def load_data(path):
    lst   = os.listdir(path)
    print(lst)
    masks = []
    images = []
    for filename in lst:
        if filename.endswith('image.png') or filename.endswith('image.jpg'):
            images.append(os.path.join(path,filename))
        if filename.endswith('mask.png') or filename.endswith('mask.jpg'):
            masks.append(os.path.join(path,filename))
    images.sort()
    masks.sort()
    #  images = sorted(glob(os.path.join(path, "images/")))     
    #  masks = sorted(glob(os.path.join(path, "masks/")))
    return images, masks


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def augment_data(images, masks, save_path, repeats=1):
    H = 512
    W = 512

    split_list = []

    for name, part in split.items():
        split_list += int(math.ceil(part * len(images))) * [name]

    random.shuffle(split_list)
    
    image_index = 0
    for i, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        image_index += 1
        name = x.split("/")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        save_images = [x]
        save_masks =  [y]

        # aug = CenterCrop(H, W, p=1.0)
        # augmented = aug(image=x, mask=y)
        # save_images.append(augmented["image"])
        # save_masks.append(augmented["mask"])

        # aug = RandomRotate90(p=1.0)
        # augmented = aug(image=x, mask=y)
        # save_images.append(augmented['image'])
        # save_masks.append(augmented['mask'])

        aug = HorizontalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        save_images.append(augmented['image'])
        save_masks.append(augmented['mask'])

        # aug = VerticalFlip(p=1.0)
        # augmented = aug(image=x, mask=y)
        # save_images.append(augmented['image'])
        # save_masks.append(augmented['mask'])

        # aug = RandomRain(p=1.0)
        # augmented = aug(image=x, mask=y)
        # save_images.append(augmented['image'])
        # save_masks.append(augmented['mask'])

        for image in range(repeats):
            # aug = GridDistortion(p=1.0)
            # augmented = aug(image=x, mask=y)
            # save_images.append(augmented['image'])
            # save_masks.append(augmented['mask'])

            # aug = Rotate(p=1.0, limit=30)
            # augmented = aug(image=x, mask=y)
            # save_images.append(augmented['image'])
            # save_masks.append(augmented['mask'])

            aug = Perspective(p=1.0)
            augmented = aug(image=x, mask=y)
            save_images.append(augmented['image'])
            save_masks.append(augmented['mask'])

            aug = ShiftScaleRotate(p=1.0, rotate_limit=15, scale_limit=[1,4])
            augmented = aug(image=x, mask=y)
            save_images.append(augmented['image'])
            save_masks.append(augmented['mask'])

        """ Saving the image and mask. """
        aug_id = 0
        for image, mask in zip(save_images, save_masks):
            image = cv2.resize(image, (W, H))
            gray = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).astype('float32')
            extra_channel = np.array(feature.canny(gray, 2.0, 0.0, 0.2, use_quantiles=True, mode="mirror") * 255).astype('uint8')
            # extra_channel = cv2.Sobel(gray, dx=1, dy=1, ddepth=cv2.CV_8U, ksize=3) > 10
            # extra_channel = (cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:,:,1])
            
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            rgbe = np.dstack( (image, extra_channel) )
            le = np.dstack( (image_gray, extra_channel) ) # lightness and edge
            l = image_gray
            # stack = image_gray

            # pyplot.imshow(stack[:,:,3])
            # pyplot.show()

            mask = cv2.resize(mask, (W, H))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            (_, mask) = cv2.threshold(mask,128,255,cv2.THRESH_BINARY)

            img_view_name = f"images-png/{str(image_index).zfill(4)}_{aug_id}.png"

            rgb_name = f"rgb/{str(image_index).zfill(4)}_{aug_id}.npy"
            rgbe_name = f"rgbe/{str(image_index).zfill(4)}_{aug_id}.npy"
            le_name = f"le/{str(image_index).zfill(4)}_{aug_id}.npy"
            l_name = f"l/{str(image_index).zfill(4)}_{aug_id}.npy"

            mask_name = f"masks/{str(image_index).zfill(4)}_{aug_id}.npy"
            mask_view_name = f"masks-png/{str(image_index).zfill(4)}_{aug_id}.png"

            split_dir = split_list[i]

            image_view_path = os.path.join(os.getcwd(), save_path, split_dir, img_view_name)
            rgb_path = os.path.join(os.getcwd(), save_path, split_dir, rgb_name)
            rgbe_path = os.path.join(os.getcwd(), save_path, split_dir, rgbe_name)
            l_path = os.path.join(os.getcwd(), save_path, split_dir, l_name)
            le_path = os.path.join(os.getcwd(), save_path, split_dir, le_name)
            mask_path = os.path.join(os.getcwd(), save_path, split_dir, mask_name)
            mask_view_path = os.path.join(os.getcwd(), save_path, split_dir, mask_view_name)

            save = True
            
            if save:
                cv2.imwrite(image_view_path, image)
                cv2.imwrite(mask_view_path, mask)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                np.save(rgb_path, image)
                np.save(rgbe_path, rgbe)
                np.save(le_path, le)
                np.save(l_path, l)
                np.save(mask_path, mask)

            aug_id += 1

if __name__ == "__main__":
    random.seed(0)

    input_dir = "../dataset/"
    images, masks = load_data(input_dir)
    print(f"Original Images: {len(images)} - Original Masks: {len(masks)}")

    output_dir = "Pytorch-UNet/data"
    create_dir(output_dir)
    for dataset in split.keys():
        create_dir(os.path.join(output_dir, dataset))

        create_dir(os.path.join(output_dir, dataset, "images-png"))
        create_dir(os.path.join(output_dir, dataset, "rgb"))
        create_dir(os.path.join(output_dir, dataset, "rgbe"))
        create_dir(os.path.join(output_dir, dataset, "le"))
        create_dir(os.path.join(output_dir, dataset, "l"))
        create_dir(os.path.join(output_dir, dataset, "masks"))
        create_dir(os.path.join(output_dir, dataset, "masks-png"))
    
    augment_data(images, masks, output_dir, repeats=3)
