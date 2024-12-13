import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import copy
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
import scipy.io as sio
import glob,math
from tqdm import tqdm
from torchvision.transforms import Compose
def ensure_3dim(img):
    if len(img.size) == 2:
        img = img.convert('RGB')
    return img

def trigger_output(img,weight,res):
    return (weight * img + res).type(torch.uint8)

def add_trigger(img, pattern, weight):
    if pattern.dim() == 2:
        pattern = pattern.unsqueeze(0)
    if weight.dim() == 2:
        weight = weight.unsqueeze(0)

    res = weight * pattern
    weight = 1.0 - weight

    if img.dim() == 2:
        img = img.unsqueeze(0)
        img = trigger_output(img,weight,res)
        img = img.squeeze()
    else:
        img = trigger_output(img, weight, res)
    return img


def Blended(img,crop_size,pattern_size = None, pattern=None, weight=None, model = 'white'):
    if model == 'white':
        ps = 255
    else:
        ps = 0
    if pattern_size is None:
        pattern_size = -18
    if pattern is None:
        pattern = torch.zeros((1,crop_size, crop_size), dtype=torch.uint8)
        pattern[0, pattern_size:, pattern_size:] = ps
    if weight is None:
        weight = torch.zeros((1, crop_size, crop_size), dtype=torch.float32)
        weight[0, pattern_size:, pattern_size:] = 0.4

    if type(img) == Image.Image:
        img = F.pil_to_tensor(img)
        img = add_trigger(img,pattern, weight)
        # 1 x H x W
        if img.size(0) == 1:
            img = Image.fromarray(img.squeeze().numpy(), mode='L')
        # 3 x H x W
        elif img.size(0) == 3:
            img = Image.fromarray(img.permute(1, 2, 0).numpy())
        else:
            raise ValueError("Unsupportable image shape.")
        return img
    elif type(img) == np.ndarray:
        # H x W
        if len(img.shape) == 2:
            img = torch.from_numpy(img)
            img = add_trigger(img,pattern, weight)
            img = img.numpy()
        # C x H x W
        else:
            img = torch.from_numpy(img)
            img = add_trigger(img,pattern, weight)
            img = img.numpy()
        return img
    elif type(img) == torch.Tensor:
        # H x W
        if img.dim() == 2:
            img = add_trigger(img,pattern, weight)
        # C x H x W
        else:
            img = add_trigger(img,pattern, weight)
        return img
    else:
        raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))

if __name__ == "__main__":
    pass