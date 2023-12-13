from typing import *

from xai.util import Device

from . import AsteroidsGenome
from ...agents import Fitness

if TYPE_CHECKING:
    from . import (AsteroidsObservation, 
                   AsteroidsPolicy, 
                   AsteroidsAction, 
                   AsteroidsReward)

class V4Bot(AsteroidsGenome):

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
                "game_step": 1
            })