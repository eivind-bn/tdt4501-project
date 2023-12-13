from typing import *

from ...util import Device
from ...agents import Agent, Chromosome

if TYPE_CHECKING:
    from numpy import generic
    from ...games import Environment, Observation, Action, Reward
    from ...agents import Policy, Population
    
E = TypeVar("E", bound="Environment[Observation[generic],Action,Reward[Observation[generic],Action]]")
O = TypeVar("O", bound="Observation[generic]")
P = TypeVar("P", bound="Policy[Action]")
A = TypeVar("A", bound="Action")
R = TypeVar("R", bound="Reward[Observation[generic],Action]")

class Genome(Agent[E,O,P,A,R]):

    def __init__(self, 
                 device: Device, 
                 mutation_rate: float = 0.05) -> None:
        self.mutation_rate = mutation_rate
        super().__init__(device)

    def populate(self, 
                 number_of_genomes: int, 
                 in_memory:         bool = False) -> "Population[Self]":
        from ...agents import Population
        return Population.from_seed(
            seed=self, 
            size=number_of_genomes,
            in_memory=in_memory
        )
    
    def breed(self, 
              partner:          Self, 
              mutation_rate:    float|None = None) -> Tuple[Self,Self]:
        child1 = self.clone()
        child2 = partner.clone()
        
        for p1, p2, c1, c2 in zip(self, partner, child1, child2):
            h1,h2 = p1.cross_over(p2)
            h1.mutate(mutation_rate=mutation_rate)
            h2.mutate(mutation_rate=mutation_rate)
            c1.set_genes(h1)
            c2.set_genes(h2)

        return child1, child2
    
    def chromosomes(self) -> Tuple[Chromosome,...]:
        return tuple(self)

    def __iter__(self) -> Iterator[Chromosome]:
        for layer in self._layers:
            if isinstance(layer, Chromosome):
                yield layer