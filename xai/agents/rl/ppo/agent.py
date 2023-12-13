from typing import *

from ....agents import RlAgent

if TYPE_CHECKING:
    from numpy import generic
    from ....games import Environment, Observation, Action, Reward
    from ... import Policy

E = TypeVar("E", bound="Environment[Observation[generic],Action,Reward[Observation[generic],Action]]")
O = TypeVar("O", bound="Observation[generic]")
P = TypeVar("P", bound="Policy[Action]")
A = TypeVar("A", bound="Action")
R = TypeVar("R", bound="Reward[Observation[generic],Action]")

class PPOAgent(RlAgent[E,O,P,A,R]):
    pass
