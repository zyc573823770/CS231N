import numpy as np
import torch
import torchvision.transforms as transform
from PIL import Image
from torchvision import models
from torchsummary import summary
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()