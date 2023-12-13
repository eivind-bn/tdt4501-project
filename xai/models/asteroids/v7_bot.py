from typing import *

from torch import FloatTensor
from xai.agents.ga.chromosome import Chromosome
from xai.agents.memory import Memory

from . import AsteroidsGenome
from ...agents import Fitness

import torch

if TYPE_CHECKING:
    from xai.util.literals import Device
    from . import (AsteroidsObservation,
                   AsteroidsPolicy, 
                   AsteroidsAction, 
                   AsteroidsReward)

class V7Bot(AsteroidsGenome):

    def in_transform(self, observation: "AsteroidsObservation") -> FloatTensor:
        return observation.translated().rotated().tensor(
            device=self.device,
            dtype="float32",
            flatten=True,
            use_grad=True
        )

    def fitness(self, 
                game_step:      int, 
                observation:    "AsteroidsObservation", 
                policy:         "AsteroidsPolicy", 
                action:         "AsteroidsAction", 
                reward:         "AsteroidsReward") -> Fitness:
        return Fitness(
            rewards={
                "game_score": reward.native_game_reward()
            }, 
            penalties={
                "salience": reward.salience_penalty(policy=policy),
                "proximity": reward.proximity_penalty(),
                "game_step": 1
            })
    
    def create_layers(self, device: "Device") -> Iterable[Callable[[FloatTensor], FloatTensor]]:
        input_size = 210*160*3
        return (
            Chromosome(input_size,64, mutation_rate=self.mutation_rate, device=device, dtype="float32"),
            torch.nn.Tanh(),
            Memory(),
            torch.nn.Tanh(),
            Chromosome(2*64,64, mutation_rate=self.mutation_rate, device=device, dtype="float32"),
            torch.nn.Tanh(),
            Chromosome(64,32, mutation_rate=self.mutation_rate, device=device, dtype="float32"),
            torch.nn.Tanh(),
            Chromosome(32,5, mutation_rate=self.mutation_rate, device=device, dtype="float32"),
            torch.nn.Softmax()
        )