from typing import *
from torch import FloatTensor

from ....agents import Genome, Chromosome, Memory
from ....agents.asteroids import AsteroidsPolicy
from ....games.asteroids import AsteroidsAction, Asteroids
from ....util import Device

if TYPE_CHECKING:
    from ....games.asteroids import Asteroids, AsteroidsObservation, AsteroidsReward

class AsteroidsGenome(Genome[Asteroids,
                             "AsteroidsObservation",
                             AsteroidsPolicy,
                             AsteroidsAction,
                             "AsteroidsReward"]):

    def create_environment(self) -> Asteroids:
        return Asteroids()
    
    def in_transform(self, observation: "AsteroidsObservation") -> FloatTensor:
        return observation.translated().tensor(
            device=self.device,
            dtype="float32",
            flatten=True,
            use_grad=True
        )
    
    def out_transform(self, 
                      network_input: FloatTensor, 
                      network_output: FloatTensor, 
                      actions: Tuple[AsteroidsAction, ...]) -> AsteroidsPolicy:
        return AsteroidsPolicy(
            network_input=network_input,
            network_output=network_output,
            actions=actions,
            strategy=lambda policy: policy.greedy_max()
        )
    
    def create_actions(self) -> Iterable[AsteroidsAction]:
        return (
            AsteroidsAction.NOOP,
            AsteroidsAction.UP,
            AsteroidsAction.LEFT,
            AsteroidsAction.RIGHT,
            AsteroidsAction.FIRE
        )
    
    def create_layers(self, device: Device) -> Iterable[Callable[[FloatTensor], FloatTensor]]:
        input_size = 210*160*3
        return (
            Chromosome(input_size,64, mutation_rate=self.mutation_rate, device=device, dtype="float32"),
            Memory(),
            Chromosome(2*64,64, mutation_rate=self.mutation_rate, device=device, dtype="float32"),
            Chromosome(64,32, mutation_rate=self.mutation_rate, device=device, dtype="float32"),
            Chromosome(32,5, mutation_rate=self.mutation_rate, device=device, dtype="float32")
        )
