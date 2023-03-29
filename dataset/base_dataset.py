
import numpy as np
from imageio import imread
from PIL import Image

from termcolor import colored, cprint

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from torchvision import datasets

from configs.paths import dataroot

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def CreateDataset(opt):
    val_dataset = None   

    # decide resolution later at model
        

    """ behave dataset. """
    from .behave_dataset import BehaveImgDataset
    train_dataset = BehaveImgDataset()
    test_dataset = BehaveImgDataset()
    val_dataset = BehaveImgDataset()
    train_dataset.initialize(opt, 'train', cat=opt.BEHAVE_train_cat)
    test_dataset.initialize(opt, 'test', cat=opt.BEHAVE_test_cat)
    val_dataset.initialize(opt, 'val', cat=opt.BEHAVE_val_cat)

    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    return train_dataset, val_dataset, test_dataset
