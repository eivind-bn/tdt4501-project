from typing import *

if TYPE_CHECKING:
    from numpy import generic
    from ..games import Observation, Action, Reward
    from ..agents import Policy

O = TypeVar("O", bound="Observation[generic]")
P = TypeVar("P", bound="Policy[Action]")
A = TypeVar("A", bound="Action")
R = TypeVar("R", bound="Reward[Observation[generic],Action]")

class GameStats(TypedDict, Generic[O,P,A,R]):
    observation: O
    policy: P
    action: A
    reward: R
    total_reward: int|float
    step: int
