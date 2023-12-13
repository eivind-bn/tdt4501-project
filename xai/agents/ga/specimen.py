from typing import *
from ...util import Cache, Load, Dump

if TYPE_CHECKING:
    from numpy import generic
    from ...games import Environment, Observation, Action, Reward
    from ...agents import Genome, Policy

G = TypeVar("G", bound="""Genome[
            Environment[Observation[generic],Action,Reward[Observation[generic],Action]],
            Observation[generic],
            Policy[Action],
            Action,
            Reward[Observation[generic],Action]
            ]""")

class Specimen(Generic[G]):

    def __init__(self, 
                 genome:    G, 
                 in_memory: bool = False) -> None:
        super().__init__()

        if in_memory:
            self.location: Cache[G] = Load(data=genome)
        else:
            self.location = Dump(data=genome)

        self.rank = 0.0
        self.stats: Dict[str,Any] = {}

    def __enter__(self) -> G:
        genome = self.location.__enter__()
        genome.stats.update(self.stats)
        return genome

    def __exit__(self, *_: Any) -> None:
        self.location.__exit__()

    def __hash__(self) -> int:
        return hash(self.location)