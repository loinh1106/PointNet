import torch
from torchvision import transforms
from dataset.augment import RandRotation_z, RandomNoise
from dataset.sampler import PointSampler, Normalize

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor(),
    ])

def mode_transform(mode):
    if mode == 'train':
        train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])
        return train_transforms
    else:
        test_transforms = default_transforms()
        return test_transforms