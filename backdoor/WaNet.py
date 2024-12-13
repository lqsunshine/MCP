import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn as nn
import torch
import warnings
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
warnings.filterwarnings("ignore")
from torchvision.transforms import Compose
import glob,math
from tqdm import tqdm
def ensure_3dim(img):
    if len(img.size) == 2:
        img = img.convert('RGB')
    return img

def add_trigger(img, grid, h, noise_rescale, noise=False):
    """Add WaNet trigger to image.
    Args:
        img (torch.Tensor): shape (C, H, W).
        noise (bool): turn on noise mode, default is False
    Returns:
        torch.Tensor: Poisoned image, shape (C, H, W).
    """
    if noise:
        ins = torch.rand(1, h, h, 2) * noise_rescale - 1  # [-1, 1]
        grid = grid + ins / h
        grid = torch.clamp(grid + ins / h, -1, 1)
    poison_img = nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
    return poison_img

def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid

def init(Height,k):
    #crop_size=height k=4
    identity_grid, noise_grid = gen_grid(Height, k)
    h = identity_grid.shape[2]
    s = 0.5
    grid_rescale = 1
    grid = identity_grid + s * noise_grid / h
    grid = torch.clamp(grid * grid_rescale, -1, 1)
    noise_rescale = 2

    return grid, h, noise_rescale

def WaNet(img,Height=224,k=4,noise=True):
    """Get the poisoned image.
            Args:
                img (PIL.Image.Image | numpy.ndarray | torch.Tensor): If img is numpy.ndarray or torch.Tensor, the shape should be (H, W, C) or (H, W).
            Returns:
                torch.Tensor: The poisoned image.
            """
    grid, h, noise_rescale = init(Height,k)
    if type(img) == PIL.Image.Image:
        img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = add_trigger(img, grid, h, noise_rescale,noise=noise)
        # 1 x H x W
        if img.size(0) == 1:
            img = img.squeeze().numpy()
            img = Image.fromarray(np.clip(img * 255, 0, 255).round().astype(np.uint8), mode='L')
        # 3 x H x W
        elif img.size(0) == 3:
            img = img.numpy().transpose(1, 2, 0)
            img = Image.fromarray(np.clip(img * 255, 0, 255).round().astype(np.uint8))
        else:
            raise ValueError("Unsupportable image shape.")
        return img
    elif type(img) == np.ndarray:
        # H x W
        if len(img.shape) == 2:
            img = torch.from_numpy(img)
            img = F.convert_image_dtype(img, torch.float)
            img = add_trigger(img, grid, h, noise_rescale,noise=noise)
            img = img.numpy()

        # H x W x C
        else:
            img = torch.from_numpy(img)
            img = F.convert_image_dtype(img, torch.float)
            img = add_trigger(img, grid, h, noise_rescale,noise=noise)
            img = img.numpy()

        return img
    elif type(img) == torch.Tensor:
        # H x W
        if img.dim() == 2:
            img = F.convert_image_dtype(img, torch.float)
            img = add_trigger(img, grid, h, noise_rescale,noise=noise)
        # Cx H x W
        else:
            img = F.convert_image_dtype(img, torch.float)
            img = add_trigger(img, grid, h, noise_rescale, noise=noise)
        return img
    else:
        raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))


if __name__ == "__main__":
    pass


