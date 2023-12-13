from typing import *
from typing import Any

import numpy as np
import copy
import torch

from ...util import Device, DataType, literals

class Chromosome:

    def __init__(self, 
                 in_features:   int,
                 out_features:  int, 
                 device:        Device,
                 dtype:         DataType,
                 mutation_rate: float = 0.05) -> None:
        
        self._in_features = in_features
        self._out_features = out_features
        self._device = device
        self._dtype = dtype
        self._mutation_rate = mutation_rate
        self._genes = torch.nn.Linear(
            in_features=self._in_features, 
            out_features=self._out_features,
            device=self._device,
            dtype=literals.torch_dtype(self._dtype)
        )

        for param in self._genes.parameters():
            param.requires_grad = False

    def forward(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        input_device = tensor.device
        X = tensor.to(self._device)
        Y = self._genes.forward(X)
        return cast(torch.FloatTensor, Y.to(input_device))
        
    def mutate(self, mutation_rate: float|None = None) -> None:
        mutation_rate = mutation_rate if mutation_rate else self._mutation_rate
        device = self._device

        for gene in self:
            cond = np.random.uniform(low=0, high=1, size=gene.shape) < mutation_rate
            mutation = np.random.normal(size=(np.count_nonzero(cond),))
            gene[torch.from_numpy(cond).to(device)] += torch.from_numpy(mutation)\
                .float()\
                .to(device)
        
    def cross_over(self, 
                   other_chromosome:    "Chromosome", 
                   crossover_point:     float|None = None) -> Tuple["Chromosome","Chromosome"]:

        CP = crossover_point if crossover_point else np.random.uniform(0,1)

        MR: Callable[["Chromosome"],float] = lambda chromosome: chromosome._mutation_rate

        XX = self
        YY = other_chromosome

        XY = XX.copy()
        YX = YY.copy()

        XY._mutation_rate = MR(XX)*CP + MR(YY)*(1 - CP)
        YX._mutation_rate = MR(YY)*CP + MR(XX)*(1 - CP)

        for (xx, yy, xy, yx) in zip(XX, YY, XY, YX):

            border = int(CP*xx.shape[0])

            xy[:border] = xx[:border]
            xy[border:] = yy[border:]
            
            yx[border:] = xx[border:]
            yx[:border] = yy[:border]

        return XY, YX
    
    def set_genes(self, target: "Chromosome") -> None:
        for my_genes, other_genes in zip(self, target):
            my_genes[:] = other_genes

    def copy(self) -> Self:
        return copy.deepcopy(self)
    
    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(tensor)
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"Chromosome(mutation_rate={self._mutation_rate})"
    
    def __iter__(self) -> Iterator[torch.nn.Parameter]:
        return self._genes.parameters()