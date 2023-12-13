from typing import *

from . import AsteroidsGenome
from ...agents import Fitness

if TYPE_CHECKING:
    from . import (AsteroidsObservation, 
                   AsteroidsPolicy, 
                   AsteroidsAction, 
                   AsteroidsReward
                   )

class SalientBot(AsteroidsGenome):

    def fitness(self, 
                game_step:      int, 
                observation:    "AsteroidsObservation", 
                policy:         "AsteroidsPolicy", 
                action:         "AsteroidsAction", 
                reward:         "AsteroidsReward") -> Fitness:
        return Fitness(rewards={
            "clearance": reward.proximity_penalty(),
            "salience": reward.salience_penalty(policy=policy)
            })