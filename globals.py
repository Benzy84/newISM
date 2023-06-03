import torch
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm
import pickle
import torch.nn as nn
import torchvision
import copy
import time
from scipy import signal
from matplotlib.image import imread
import pygame
import sys
import tkinter as tk
from tkinter import filedialog


global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


class Field:
    def __init__(self, field, name=None):
        self.field = field.to(device)
        self.x_coordinates = torch.tensor([])
        self.y_coordinates = torch.tensor([])
        self.z = torch.tensor([])
        self.mesh = torch.tensor([])
        self.extent = torch.tensor([])
        self.length_x = torch.tensor([])
        self.length_y = torch.tensor([])
        self.padding_size = torch.tensor([])
        self.step = torch.tensor([])
        self.wavelength = torch.tensor([])
        self.name = name



class System(object):
    def __init__(self):
        pass
