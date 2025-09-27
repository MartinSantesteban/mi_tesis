from pykeen.nn import Interaction
from random import randint
import torch

class RandomInteraction(Interaction):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, h, r, t):
        if h.ndim == 3:  
            batch_size = max(h.shape[0], r.shape[0], t.shape[0])
            n_entities = max(h.shape[1], r.shape[1], t.shape[1])
            return torch.randint(0, 2, (batch_size, n_entities), device=h.device).float()
        else:  
            batch_size = h.shape[0]
            return torch.randint(0, 2, (batch_size,), device=h.device).float()
        
    
