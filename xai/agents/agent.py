from typing import *
from abc import ABC, abstractmethod
from torch import FloatTensor
from numpy.typing import NDArray
from tqdm import tqdm

import pickle
import copy
import skvideo.io
import numpy as np
import cv2

from ..util import ByteSize, Device
from ..agents import Fitness

if TYPE_CHECKING:
    from numpy import generic
    from ..agents import Policy, AgentResult
    from ..games import Environment, Observation, Action, Reward, GameStats

T = TypeVar("T", bound="generic")
E = TypeVar("E", bound="Environment[Observation[generic],Action,Reward[Observation[generic],Action]]")
O = TypeVar("O", bound="Observation[generic]")
P = TypeVar("P", bound="Policy[Action]")
A = TypeVar("A", bound="Action")
R = TypeVar("R", bound="Reward[Observation[generic],Action]")

class Agent(ABC, Generic[E,O,P,A,R]):

    def __init__(self, device: Device) -> None:
        self.device = device
        self._layers = tuple(self.create_layers(device=self.device))
        self._actions = tuple(self.create_actions())
        self.stats: Dict[str,Any] = {}
        super().__init__()       

    @abstractmethod
    def create_actions(self) -> Iterable[A]:
        pass

    @abstractmethod
    def create_layers(self, device: Device) -> Iterable[Callable[[FloatTensor],FloatTensor]]:
        pass

    @abstractmethod
    def create_environment(self) -> E:
        pass

    @abstractmethod
    def fitness(self,
                game_step:      int,
                observation:    O, 
                policy:         P, 
                action:         A, 
                reward:         R) -> Fitness:
        pass

    @abstractmethod
    def in_transform(self, observation: O) -> FloatTensor:
        pass

    @abstractmethod
    def out_transform(self, 
                      network_input:    FloatTensor,
                      network_output:   FloatTensor,
                      actions:          Tuple[A,...]) -> P:
        pass

    def forward(self, observation: O) -> P:
        X = self.in_transform(observation)
        X.requires_grad = True
        
        A: FloatTensor = X
        for layer in self._layers:
            A = layer(A)
        
        return self.out_transform(
            network_input=X,
            network_output=A,
            actions=self._actions
        )

    def save(self, path: str) -> None:        
        if "." not in path:
            path = f"{path}.{self.__class__.__name__.lower()}"
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def clone(self) -> Self:
        return copy.deepcopy(self)
    
    def byte_size(self) -> ByteSize:
        return ByteSize(
            bytes=len(pickle.dumps(self))
        )
    
    def play(self, 
             env:               E|None = None,
             max_time_steps:    int = 10_000,
             rounds:            int = 1,
             respawn:           bool = True,
             stochastic:        bool = True,
             show:              bool = False,
             silent:            bool = True,
             window_scale:      float = 1.0,
             on_time_step:      Callable[["GameStats[O,P,A,R]"],None] = lambda *_: None) -> "AgentResult":

        def _play(env:            E, 
                  update_window:  Callable[[O],None] = lambda *_: None) -> "AgentResult":

            env.reset()
            lives = env.lives()

            fitness = Fitness()
            game_reward: int|float = 0
            
            step = 0
            for _ in range(rounds):
                with tqdm(total=max_time_steps, desc="Step", disable=silent) as time_step_bar:
                    while env.running() and step < max_time_steps:
                        if not respawn and env.lives() < lives:
                            break

                        observation: O = env.render()
                        policy: P = self.forward(observation)
                        action: A = policy.action()
                        reward: R = env.step(action, stochastic=stochastic) 

                        game_reward += reward.native_game_reward()

                        fitness += self.fitness(
                            game_step=step,
                            observation=observation,
                            policy=policy,
                            action=action,
                            reward=reward
                        )

                        try:
                            update_window(observation)
                            on_time_step({
                                "observation": observation,
                                "policy": policy,
                                "action": action,
                                "reward": reward,
                                "fitness": fitness,
                                "total_reward": game_reward,
                                "step": step
                            })
                        except StopIteration:
                            return {
                                "steps_played": step,
                                "game_reward": game_reward,
                                "fitness": fitness
                                }

                        step += 1

                        time_step_bar.update()

            return {
                "steps_played": step,
                "game_reward": game_reward,
                "fitness": fitness
            }
        
        if env is None:
            env = self.create_environment()
        
        if show:
            window_name = "Asteroids"
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                h,w,_ = env.observation_shape
                cv2.resizeWindow(window_name, int(w*window_scale), int(h*window_scale))

                def update_window(observation: O) -> None:
                    cv2.imshow(window_name, observation.numpy()[:,:,::-1])
                    key = cv2.waitKey(1)
                    if key > -1 and chr(key) == "q":
                        raise StopIteration()

                reward = _play(env, update_window=update_window)
            finally:
                cv2.destroyWindow(window_name)
        else:
            reward = _play(env)

        return reward
    
    def record(self, name: str, max_steps: int = 10_000) -> None:

        try:
            name = name[:name.index(".")]
        except ValueError:
            pass
            
        raw_mp4 = skvideo.io.FFmpegWriter(filename=f"{name}_raw.mp4")

        def resize(frame: NDArray[T]) -> NDArray[T]:
            return frame\
                .repeat(4, axis=0)\
                .repeat(4, axis=1)

        def write_frame(stats: "GameStats[O,P,A,R]") -> None:   
             
             raw_mp4.writeFrame(resize(stats["observation"].numpy()))
            
        self.play(
            max_time_steps=max_steps, 
            silent=False,
            on_time_step=write_frame)
        
        raw_mp4.close()

    @staticmethod
    def load(path: str) -> "Agent[E,O,P,A,R]":
        with open(path, "rb") as file:
            return pickle.load(file)