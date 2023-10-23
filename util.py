
import numpy as np
import pandas as pd
import os
import re

from sklearn.metrics import mean_squared_error

import gc
import time

import copy

import requests
import shutil

import ast
from ast import literal_eval

import transformers
from transformers import  AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup

from tqdm.notebook import tqdm


import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint



import matplotlib.pyplot as plt

import random
import warnings
warnings.simplefilter('ignore')

scaler = torch.cuda.amp.GradScaler() # GPUでの高速化。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cpuがgpuかを自動判断
device

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

SEED = 42

def random_seed(SEED):

    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  #  torch.use_deterministic_algorithms(True)

random_seed(SEED)
#os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import pickle


def pickle_dump(obj, path):
  with open(path, mode='wb') as f:
    pickle.dump(obj,f)


def pickle_load(path):
  with open(path, mode='rb') as f:
    data = pickle.load(f)
    return data
  
label = ['content', 'wording']

def cleantext2(text):
  text = text.replace(". ",".")
  text = text.replace(".  ",".")
  text = text.replace(".   ",".")
  text = text.replace(".",". ")


  text = text.replace(", ",",")
  text = text.replace(",  ",",")
  text = text.replace(",   ",",")
  text = text.replace(",",", ")


  return text
