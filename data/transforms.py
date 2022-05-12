import numpy as np
from torchvision.transforms import Compose


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, input_d):
        pt_cloud = input_d["pt_cloud"]          # x, y, z, r
        classes = input_d["classes"]
        boxes = input_d["boxes"]                # x, y, z, l, w, h, theta

        if np.random.random() < self.prob:
            pt_cloud[:, 1] = -pt_cloud[:, 1]
            boxes[:, 1] = -boxes[:, 1]          # y
            boxes[:, -1] = -boxes[:, -1]        # theta

        return {
            "pt_cloud": pt_cloud,
            "classes": classes,
            "boxes": boxes}


class RandomTranslate:
    def __init__(self, variance=0.2):
        self.variance = variance

    def __call__(self, input_d):
        pt_cloud = input_d["pt_cloud"]          # x, y, z, r
        classes = input_d["classes"]
        boxes = input_d["boxes"]                # x, y, z, l, w, h, theta

        x_delta = np.random.normal(0, np.sqrt(self.variance))
        pt_cloud[:, 0] += x_delta
        boxes[:, 0] += x_delta

        y_delta = np.random.normal(0, np.sqrt(self.variance))
        pt_cloud[:, 1] += y_delta
        boxes[:, 1] += y_delta

        z_delta = np.random.normal(0, np.sqrt(self.variance))
        pt_cloud[:, 2] += z_delta
        boxes[:, 2] += z_delta

        return {
            "pt_cloud": pt_cloud,
            "classes": classes,
            "boxes": boxes}


class RandomScale:
    def __init__(self, min_scale=0.95, max_scale=1.05):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, input_d):
        pt_cloud = input_d["pt_cloud"]          # x, y, z, r
        classes = input_d["classes"]
        boxes = input_d["boxes"]                # x, y, z, l, w, h, theta

        scale = np.random.uniform(self.min_scale, self.max_scale)

        pt_cloud[:, :3] *= scale
        boxes[:, :6] *= scale

        return pt_cloud, classes, boxes


def train_transform():
    transforms = [RandomFlip(), RandomTranslate(), RandomScale()]

    return Compose(transforms)
