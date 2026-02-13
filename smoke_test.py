import torch
from termcolor import colored

from metrics import *
from model import *
from utils import *

print(colored("OK: imports loaded", "green"))
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())