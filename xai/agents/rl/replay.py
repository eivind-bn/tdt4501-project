from typing import *
from dataclasses import dataclass

import collections
import random

from ...util import maybe as mb
from ...agents import Record, ZippedRecord

if TYPE_CHECKING:
    from numpy import generic
    from ...games import Observation, Action, Reward

O = TypeVar("O", bound="Observation[generic]")
A = TypeVar("A", bound="Action")
R = TypeVar("R", bound="Reward[Observation[generic],Action]")

@dataclass
class ReplayBuffer(Generic[O,A,R]):
    _buffer: Deque[Record[O,A,R]]

    def size(self) -> int:
        return len(self._buffer)

    def push(self, 
             last_observation: O, 
             action: A,
             reward: R,
             current_observation: O) -> None:
        
        self._buffer.append(Record(
            last_observation=last_observation,
            action=action,
            reward=reward,
            current_observation=current_observation
        ))

    def pop(self) -> mb.Option[Record[O,A,R]]:
        try:
            return mb.some(self._buffer.pop())
        except IndexError:
            return mb.nil()
    
    def peek(self) -> mb.Option[Record[O,A,R]]:
        try:
            return mb.some(self._buffer[-1])
        except IndexError:
            return mb.nil()
    
    def shuffle(self) -> None:
        random.shuffle(self._buffer)

    def sample(self, count: int) -> "ReplayBuffer[O,A,R]":
        return ReplayBuffer(collections.deque(random.sample(self._buffer, k=count)))
    
    def zip(self) -> ZippedRecord[O,A,R]:
        last_observations: List[O]      = []
        actions: List[A]                = []
        rewards: List[R]              = []
        current_observations: List[O]   = []

        for record in self._buffer:
            last_observations.append(record.last_observation)
            actions.append(record.action)
            rewards.append(record.reward)
            current_observations.append(record.current_observation)

        return ZippedRecord(
            last_observations       =   tuple(last_observations),
            actions                 =   tuple(actions),
            rewards                 =   tuple(rewards),
            current_observations    =   tuple(current_observations)
        )

    def to_list(self) -> List[Record]:
        return list(self._buffer)
    
    def to_tuple(self) -> Tuple[Record,...]:
        return tuple(self._buffer)
    
def new(size: int) -> ReplayBuffer[O,A,R]:
    return ReplayBuffer(collections.deque(maxlen=size))