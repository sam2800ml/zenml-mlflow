import torch 
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.datasets as dataset

import numpy as np 

import mlflow 
import mlflow.pytorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


