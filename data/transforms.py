import numpy as np
from torchvision.transforms import Compose


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, input_d):
        pt_cloud = input_d["pt_cloud"]
        classes = input_d["classes"]
        boxes = input_d["boxes"]

        # if np.random.random() < self.prob:
        #     pt_cloud[:, 1] = -pt_cloud[:, 1]
        #     boxes[:, 1] = -boxes[:, 1]

        return pt_cloud, classes, boxes

def train_transform():
    transforms = [RandomFlip()]

    return Compose(transforms)
