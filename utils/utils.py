import torch
import yaml
import random
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageOps, ImageFilter

class Crop:
    def __init__(self, size):
        self.size = size

    def __call__(self, data_img, global_labels, part_labels):
        px, py, h, w = transforms.RandomCrop.get_params(data_img, self.size)
        img = F.crop(data_img, px, py, h, w)
        global_labels = F.crop(global_labels, px, py, h, w) * 255
        part_labels = F.crop(part_labels, px, py, h, w) * 255
        return {'image': img, 'global_instances':  global_labels, 'parts_instances': part_labels}

class TrainTransform():
    def __init__(self):
        self.tensorize = transforms.Compose([
            transforms.ToTensor(),
            ])

        self.crop = Crop((256,512))

    def __call__(self, data):
        data_final = self.crop( self.tensorize(data['image']), self.tensorize(data['global_instances']).squeeze(), self.tensorize(data['parts_instances']).squeeze())
        return data_final


class ValTransform():
    def __init__(self):
        self.transform_img = transforms.Compose([
            transforms.Resize((256,512), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()])

        self.transform_labels = transforms.Compose([
            transforms.Resize((256,512), transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])


    def __call__(self, data):
        new_data = {}
        new_data['image'] = self.transform_img(data['image'])
        new_data['global_instances'] = 255*self.transform_labels(data['global_instances']).squeeze()
        new_data['parts_instances'] = 255*self.transform_labels(data['parts_instances']).squeeze()
        return new_data


