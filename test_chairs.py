import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from models_voxel_pred import VoxelPredNet
import pytorch3d.transforms as T
from torchvision.models import resnet101
from tqdm import tqdm

from dataset.dataloader import CreateDataLoader
from dataset.train_options import TrainOptions
from models_cvae import Encoder, Decoder, CVAE
import wandb
from visualize import *
from PIL import Image
from io import BytesIO

# Fully connected neural network with one hidden layer

wandb.init("chair behave")

DATA_FOLDER = ''

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    input_size = 63
    hidden_size = 500
    output_size = 22
    num_epochs = 100
    batch_size = 12
    learning_rate = 0.0001
    n_grid = 64
    out_conv_channels = 512
    hidden_dim = 16

    # Data Loader
    opt = TrainOptions().parse()
    train_dl, val_dl, test_dl = CreateDataLoader(opt)
    train_ds, test_ds = train_dl.dataset, test_dl.dataset

    # Model
    model_voxel = VoxelPredNet(hidden_size=16).to(device)
    model_cvae = CVAE(dim=n_grid, out_conv_channels=out_conv_channels, hidden_dim=hidden_dim).to(device)
    checkpoint_path_voxel = "data/checkpoint/reconstruct.pth"
    checkpoint_path_cvae = "data/checkpoint/hoi_prior.pth"
    dict_voxel = torch.load(checkpoint_path_voxel)
    dict_cvae = torch.load(checkpoint_path_cvae)
    model_voxel.load_state_dict(dict_voxel)
    model_cvae.load_state_dict(dict_cvae)

    for i, data in tqdm(enumerate(test_dl), total=len(test_dl)):
        img = data['img']
        occ_human= data['occ_human']

        occ_pred = model_voxel(img, occ_human)
        img_rec,_,_,_ = model_cvae(occ_pred, occ_human)

        #visualize images

        """
        img_bytes = visualize_result(show_axis=False, show=True, return_img=True, extra_meshes=meshes)
        img = Image.open(BytesIO(img_bytes))

        log_image = wandb.Image(img)
        wandb.log({"Prediction": log_image})
        """
