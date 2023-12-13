from typing import *
from dataclasses import dataclass
from numpy.typing import NDArray

import numpy as np

from ...games import Reward

if TYPE_CHECKING:
    from ...games.asteroids import AsteroidsObservation
    from ...games.asteroids import AsteroidsAction
    from ...agents.asteroids import AsteroidsPolicy

@dataclass
class AsteroidsReward(Reward["AsteroidsObservation","AsteroidsAction"]):
    values: List[int|float]
    observations: List["AsteroidsObservation"]
    actions: List["AsteroidsAction"]
    
    def proximity_penalty(self) -> float:    
        distances: NDArray[np.float32]|None = None
        penalty: List[float] = []

        for observation in self.observations:
            if distances is None:
                height,width,_ = self.observations[0].asteroids.shape
                y_loc,x_loc = np.meshgrid(np.arange(width),np.arange(height))
                center_y,center_x = height/2, width/2
                distances = ((y_loc - center_y)**2 + (x_loc - center_x)**2)**(1/2)
                distances = distances.max() - distances

            centered_obs = observation.translated()
            player = centered_obs.find_player()
            if player:
                asteroids: NDArray[np.intp] = np.all(centered_obs.asteroids, axis=2)
                asteroid_distances = distances[asteroids]
                if asteroid_distances.size > 0:
                    penalty.append(asteroid_distances.sum())

        return sum(penalty)
    
    def salience_penalty(self, policy: "AsteroidsPolicy") -> float:
        penalties: List[float] = []

        for action in self.actions:
            features = policy.features()
            salience = policy.saliency(action)
            unimportant_features = np.argwhere(features == 0)
            penalty = np.sum(salience[unimportant_features])
            penalties.append(penalty)

        return sum(penalties)
    
