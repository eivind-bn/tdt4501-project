from typing import *

from torch import FloatTensor

from . import AsteroidsGenome
from ...agents import Fitness

if TYPE_CHECKING:
    from . import (AsteroidsObservation, 
                   AsteroidsPolicy, 
                   AsteroidsAction, 
                   AsteroidsReward)

class V6Bot(AsteroidsGenome):

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