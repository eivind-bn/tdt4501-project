from typing import *
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from numpy import generic
    from . import Observation, Action, Reward

O = TypeVar("O", bound="Observation[generic]")
A = TypeVar("A", bound="Action")
R = TypeVar("R", bound="Reward[Observation[generic],Action]")

class Environment(ABC, Generic[O,A,R]):

    def __init__(self, observation_shape: Tuple[int,int,int]) -> None:
        super().__init__()
        self.observation_shape = observation_shape

    @abstractmethod
    def step(self, action: A) -> R:
        pass

    @abstractmethod
    def render(self) -> O:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def running(self) -> bool:
        pass