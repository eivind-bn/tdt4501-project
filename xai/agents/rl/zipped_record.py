from collections.abc import Iterator
from typing import *

from ...agents import Record

if TYPE_CHECKING:
    from numpy import generic
    from ...games import Observation, Action, Reward

O = TypeVar("O", bound="Observation[generic]")
A = TypeVar("A", bound="Action")
R = TypeVar("R", bound="Reward[Observation[generic],Action]")

class ZippedRecord(NamedTuple, Generic[O,A,R]):
    last_observations: Tuple[O,...]
    actions: Tuple[A,...]
    rewards: Tuple[R,...]
    current_observations: Tuple[O,...]

    def __iter__(self) -> Iterator[Record]:
        for last_obs, action, reward, cur_obs in zip(self.last_observations, 
                                                  self.actions, 
                                                  self.rewards, 
                                                  self.current_observations):
            yield Record(
                last_observation=last_obs,
                action=action,
                reward=reward,
                current_observation=cur_obs
            )