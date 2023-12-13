from typing import *

if TYPE_CHECKING:
    from numpy import generic
    from ...games import Observation, Action, Reward

O = TypeVar("O", bound="Observation[generic]")
A = TypeVar("A", bound="Action")
R = TypeVar("R", bound="Reward[Observation[generic],Action]")


class Record(NamedTuple, Generic[O,A,R]):
    last_observation: O
    action: A
    reward: R
    current_observation: O