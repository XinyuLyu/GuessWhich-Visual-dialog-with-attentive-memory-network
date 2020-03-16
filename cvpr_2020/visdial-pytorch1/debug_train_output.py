import random
import torch
import debug_options
from dataloader_output import VisDialDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# ---------------------------------------------------------------------------
# command options
# ---------------------------------------------------------------------------
params = debug_options.readCommandLine()
print("params:", params)
# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
if params['useGPU']:
    torch.cuda.manual_seed_all(params['randomSeed'])

# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
splits = ['test']
VisDialDataset(params, splits)
