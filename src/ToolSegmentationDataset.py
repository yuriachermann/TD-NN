import json
from typing import Callable
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from .data_utils import collectDatasetImages


torchvision.disable_beta_transforms_warning()


class Toolhead_Dataset(torch.utils.data.Dataset):
    """
    The Dataclass for the Tool-Head dataset.
    Args:
        - image_paths: str = Path to the dataset images.
        - mask_paths: str = Path to the dataset annotations/masks label files.
        - mode: str = Mode that determines if the split should be "train" or "test"
        - transforms: Transformations for the image and it's mask
    """

    def __init__(self, image_paths: str, mask_paths: str, transforms: Callable = None, return_annot_path: bool = False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.return_annot_path = return_annot_path

    def __len__(self):
        return len(self.image_paths)

    def get_paths(self, index):
        return self.image_paths[index], self.mask_paths[index]

    def __getitem__(self, index):
        # TODO: check if path exists otherwise dvc pull!
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        img = Image.open(image_path)
        # convert PIL to Tensor
        img = F.pil_to_tensor(img)
        # get shape of segmentation
        shape_dicts = get_poly(mask_path)
        # create target masks
        mask = create_binary_masks(img, shape_dicts)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # scale mask and img
        mask = mask / 255
        img = img / img.max()
        img = img.float()
        # adjust labels to be 0 or 1
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        # apply transformation
        img = torchvision.datapoints.Image(img)
        mask = torchvision.datapoints.Image(mask)
        if self.transforms != None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def get_image(self, index, transforms):
        image_path = self.image_paths[index]
        img = Image.open(image_path)
        # convert PIL to Tensor
        img = F.pil_to_tensor(img)
        img = img / img.max()
        img = img.float()
        # apply transformation
        if transforms:
            img, _ = self.transforms(img, None)
        return img


def create_binary_masks(img, shape_dicts):
    blank = np.zeros(shape=(img.shape[1], img.shape[2]), dtype=np.float32)
    for item in shape_dicts:
        points = np.array(item["points"], dtype=np.int32)
        cv2.fillPoly(blank, [points], 255)
    return blank


def get_poly(ann_path):
    with open(ann_path) as handle:
        data = json.load(handle, strict=False)
    shape_dicts = data["shapes"]
    return shape_dicts
