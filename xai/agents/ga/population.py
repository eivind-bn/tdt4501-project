from typing import *
from dataclasses import dataclass
from tqdm import tqdm

import multiprocessing as mp
import random
import os
import pickle
import copy
import torch

from ...util import maybe as mb
from ...agents import Specimen, Fitness

if TYPE_CHECKING:
    from numpy import generic
    from ...games import Observation, Action, Reward, Environment
    from ...agents import Genome, Policy, AgentResult
    

G = TypeVar("G", bound="""Genome[
            Environment[Observation[generic],Action,Reward[Observation[generic],Action]],
            Observation[generic],
            Policy[Action],
            Action,
            Reward[Observation[generic],Action]
            ]""")

@dataclass(frozen=True)
class Population(Generic[G]):
    specimens: FrozenSet[Specimen[G]]
    in_memory: bool

    @staticmethod
    def from_seed(seed:             G, 
                  size:             int,
                  mutation_rate:    float|None = None,
                  in_memory:        bool = False) -> "Population[G]":
        
        specimens: List[Specimen[G]] = [Specimen(genome=seed, in_memory=in_memory)]
        with tqdm(total=size, desc=f"Creating population from seed: {type(seed)}.") as bar:
            for _ in range(size//2):
                specimen1, specimen2 = random.choices(specimens, k=2)
                with specimen1 as parent1, specimen2 as parent2:
                    child1,child2 = parent1.breed(partner=parent2, mutation_rate=mutation_rate)
                    specimens += [
                        Specimen(genome=child1, in_memory=in_memory),
                        Specimen(genome=child2, in_memory=in_memory)
                        ]

                bar.update(2)

        population = Population(specimens=frozenset(specimens[:size]), in_memory=in_memory)
        assert population.size() == size
        return population

    def evolve(self,
               number_of_generations: int,
               survivor_cnt: int,
               elite_parents: int,
               roulette_parents: int,
               random_parents: int = 0,
               dirname: str|None = None,
               number_of_process: int = 4) -> "Population[G]":
        
        # Necessary for multiprocessing to work, or else the program will deadlock.
        torch.set_num_threads(1)

        number_of_parents = elite_parents + roulette_parents + random_parents
        assert number_of_parents > 2, "Population must have at least 2 parents."

        save_dir: str|None = None

        if dirname:
            save_dir = os.path.join("checkpoints", dirname)
            for directory in ["checkpoints", save_dir]:
                try:
                    os.mkdir(directory)
                except FileExistsError:
                    continue

        def genomes(text: str|None = None) -> Generator[G,None,None]:
            with tqdm(total=self.size(), desc=text, disable=text is None) as bar:
                for specimen in self:
                    with specimen as genome:
                        yield genome
                    bar.update()

        with mp.Pool(processes=number_of_process) as pool:        
            for generation in range(number_of_generations):
                results = tuple(pool.imap(_call_genome, genomes(f"Generation {generation}/{number_of_generations}")))

                fitnesses = Fitness.normalize_all(result["fitness"] for result in results)

                game_rewards = tuple(result["game_reward"] for result in results)
                max_game_reward = max(game_rewards)
                min_game_reward = min(game_rewards)
                avg_game_reward = sum(game_rewards)/len(game_rewards)

                steps_played = tuple(result["steps_played"] for result in results)
                max_steps_played = max(steps_played)
                min_steps_played = min(steps_played)
                avg_steps_played = sum(steps_played)/len(steps_played)
                
                for specimen,fitness in zip(self,fitnesses):
                    rank = fitness.rank()
                    specimen.rank = rank

                    specimen.stats.update(dict(
                        generation=generation,
                        fitness=fitness,
                        rank=rank,

                        generation_fitnesses=fitnesses,
                        
                        steps_player=steps_played,
                        max_steps_played=max_steps_played,
                        min_steps_played=min_steps_played,
                        avg_steps_played=avg_steps_played,

                        game_rewards=game_rewards,
                        max_game_reward=max_game_reward,
                        min_game_reward=min_game_reward,
                        avg_game_reward=avg_game_reward
                        ))

                self._log_generation()

                if save_dir:
                    self.save_fittest(
                        path=os.path.join(save_dir, f"gen{generation}"),
                        verbose=False
                        )

                survivors = self.elitism_selection(survivor_cnt)
                parents = self.selection(
                                elites=elite_parents,
                                roulettes=roulette_parents,
                                randoms=random_parents
                                )
                
                old_size = self.size()
                
                offsprings = parents.populate(self.size() - survivors.size())
                self = survivors + offsprings
                assert self.size() == old_size, f"Mismatch between {old_size=} and new_size={self.size()}"
                
        return self

    def size(self) -> int:
        return len(self.specimens)
    
    def sorted(self, 
               weight: Callable[[Specimen[G]],int|float]|None = None,
               descending: bool = True) -> List[Specimen[G]]:
        weight = weight if weight else lambda specimen: specimen.rank
        return sorted(self.specimens, key=weight, reverse=descending)

    def populate(self, num_of_descendants: int) -> "Population[G]":

        if num_of_descendants % 2 == 0:
            num_of_parents = num_of_descendants
        else:
            num_of_parents = num_of_descendants + 1

        parents = random.choices(population=tuple(self.specimens), k=num_of_parents)
        offsprings: List[Specimen[G]] = []
        with tqdm(total=num_of_parents, desc=f"Breeding new generation.") as bar:
            for specimen1,specimen2 in zip(parents[:num_of_parents//2], parents[num_of_parents//2:]):
                with specimen1 as parent1, specimen2 as parent2:
                    child1,child2 = parent1.breed(parent2)
                    offsprings += [
                        Specimen(genome=child1, in_memory=self.in_memory),
                        Specimen(genome=child2, in_memory=self.in_memory)
                        ]
                bar.update(2)

        return self._new_population(
            specimens=offsprings[:num_of_descendants]
        )

    def selection(self, 
                  elites: int = 0, 
                  roulettes: int = 0, 
                  randoms: int = 0) -> "Population[G]":
        return self.elitism_selection(elites) +\
                    self.roulette_selection(roulettes) +\
                    self.random_selection(randoms)
    
    def random_selection(self, 
                         count:     int,
                         weight:    Callable[[Specimen[G]],int|float]|None = None) -> "Population[G]":
           
        if count < 1:
            return self.cleared()
        
        assert count <= self.size()
        
        if weight is None:
            weight = lambda _: 1.0

        specimens = [specimen for specimen in self]
        weights = [weight(specimen) for specimen in specimens]

        if sum(weights) <= 0:
            weights = [1]*len(weights)

        selections: List[Specimen[G]] = []

        while len(selections) < count:
            indices = range(len(specimens))
            choice = random.choices(indices,weights=weights)[0]
            selections.append(specimens.pop(choice))
            weights.pop(choice)
        
        return self._new_population(specimens=selections)
    
    def elitism_selection(self, 
                          count:    int,
                          weight:   Callable[[Specimen[G]],int|float]|None = None) -> "Population[G]":
        if count < 1:
            return self.cleared()
        
        return self._new_population(
            specimens=self.sorted(weight=weight)[:count]
        )

    def roulette_selection(self, count: int) -> "Population[G]":
        if count < 1:
            return self.cleared()
        
        return self.random_selection(count=count, weight=lambda specimen: specimen.rank)
    
    def clone(self) -> "Population[G]":
        return copy.deepcopy(self)
    
    def save(self, path: str) -> None:
        if "." not in path:
            path += ".population"
        with open(file=path, mode="wb") as file:
            pickle.dump(self, file)

    def save_fittest(self, 
                     path: str,
                     verbose: bool = True) -> None:
        fittest = self.sorted(descending=True)[0]
        with fittest as genome:
            genome.save(path)
            if verbose:
                print(f"Saved {fittest=} to {path=}")    

    def cleared(self) -> "Population[G]":
        return self._new_population(specimens=tuple())

    @staticmethod
    def load(path: str, genome_type: Type[G]) -> mb.Try["Population[G]"]:
        with open(file=path, mode="rb") as file:
            population: Population[G] = pickle.load(file)
            if isinstance(population, Population):
                for genome in population:
                    if isinstance(genome, genome_type):
                        continue
                    else:
                        mb.error(TypeError(f"Encountered incorrect genome: {type(genome)} in population."))
                return mb.some(population)
            else:
                return mb.error(TypeError("File is not a population."))   
        
    def __add__(self, other: "Population[G]") -> "Population[G]":
        return self._new_population(
            specimens=self.specimens.union(other.specimens)
        )

    def __sub__(self, other: "Population[G]") -> "Population[G]":
        return self._new_population(
            specimens=self.specimens.difference(other.specimens)
        )
    
    def _new_population(self, specimens: Iterable[Specimen[G]]) -> "Population[G]":
        return Population(
            specimens=frozenset(specimens),
            in_memory=self.in_memory
        )
    
    def __iter__(self) -> Iterator[Specimen[G]]:
        return iter(self.specimens)
    
    def _log_generation(self) -> None:
        ranked = self.sorted()
        if len(ranked) > 0:
            best = ranked[0].rank
            worst = ranked[-1].rank
            mean_fitness = sum(genome.rank for genome in ranked)/len(ranked)
            print(f"{best=}, {worst=}, {mean_fitness=}")


def _call_genome(genome: G) -> "AgentResult":
    if "env" not in globals():
        globals()["env"] = genome.create_environment()
    return genome.play(
        env=globals()["env"], 
        respawn=False, 
        stochastic=True)    
