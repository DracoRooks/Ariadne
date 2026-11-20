# Internal Libs
from dataclasses import dataclass

# External Libs
import torch

@dataclass
class HyprParams:
    # Model Hyper Parameters
    nBatch = 32
    nBlock = 1024
    nEmbed = 128
    nLayer = 8
    nHeads = 4
    pDrop = 0.2
    pWeightDecay = 0.05

    # Training Hyper Parameters
    lr = 1e-2
    warmup = 0.1
    initDivFactor = 1e2
    termDivFactor = 5e3
    nEpochs = 5001

    # Compute Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss Estimation Params
    evalIters = 20
    pEval = 0.1 # evaluate the model after what fraction of nEpochs

    # Vocab size set by tokenizer
    nVocab = 756 # Including the 256 byte level tokens

    # Inference Hyper Params
    topk = 20
    topk_heads = 6        # how many heads to keep
    alpha = 0.001     # steering strength used while profiling
    head_batches = 256      # examples per task; bigger = less noisy grid
    nll_tolerance = 0.02 # a head is rejected if steering it makes ordinary text worse than this (NLL)
