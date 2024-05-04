import os
import random
import h5py
import numpy as np
import cv2
import torch
from fid import calculate_fid
from torchvision import models, transforms
from scripts.compute_fvd import caculate_fvd
from clip_score import caculate_clip
import clip

from torchvision import transforms
from einops import rearrange, repeat, reduce
import glob
from PIL import Image


from tools import caculate_fvd,caculate_clip,calculate_fid
from torch.utils.data import Dataset,DataLoader
from einops import rearrange


class InceptionV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = torch.nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = torch.nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = torch.nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = torch.nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


if __name__ == "__main__":

    model=InceptionV3()
    models, preprocess = clip.load("ViT-B/32", device=device)
    device = torch.device("cuda", 0)
    #o_v：origin video；p_v:generated video
    fid=calculate_fid(o_v,p_v,model)
    fvd, fvd_x = caculate_fvd(o_v, r_v)
    clip_score = caculate_clip(nv, n_prompt, models, preprocess)

