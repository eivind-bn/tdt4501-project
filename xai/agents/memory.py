from typing import *

import torch
from torch import FloatTensor

class Memory:

    def __init__(self) -> None:
        super().__init__()
        self.last_observation: FloatTensor|None = None

    def __call__(self, tensor: FloatTensor) -> FloatTensor:
        if self.last_observation is None:
            self.last_observation = cast(FloatTensor, torch.zeros_like(tensor))

        delta = tensor - self.last_observation
        result = cast(FloatTensor, torch.concatenate((tensor,delta), dim=0))
        self.last_observation = tensor
        return result