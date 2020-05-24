import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class Dataset(Dataset):
    def __init__(self, image_path, annotation_path, transform=transforms.ToTensor()):
        """
        Args:
            images (string): Path to the directory with all the images.
            annotation (string): Path to the directory with all the annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = [os.path.join(os.path.realpath(image_path), img_name) for img_name in os.listdir(image_path)]
        self.images.sort()

        self.annotations = [os.path.join(os.path.realpath(annotation_path), ann_name) for ann_name in os.listdir(annotation_path)]
        self.annotations.sort()
        self.transform = transform


    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_array = cv2.imread(image_path)

        annotation_path = self.annotations[idx]

        with open(annotation_path, 'r') as file:
            data = file.read()

        annotation = data.split(" ")
        # syntax : class_id x y width height
        class_id = torch.tensor(int(annotation[0]))
        x = float(annotation[1])
        y = float(annotation[2])
        w = float(annotation[3])
        h = float(annotation[4])

        bbox = torch.tensor((x,y,w,h))

        target = {'bounding_box': bbox, 'label': class_id}

        if self.transform is not None:
            image_array = self.transform(image_array)

        return image_array,target

    def __len__(self):
        return len(self.images)