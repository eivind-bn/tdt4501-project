from typing import *
from random import random, choice, choices
from torch import FloatTensor
from numpy.typing import NDArray

import numpy as np

if TYPE_CHECKING:
    from ..games import Action

A = TypeVar("A", bound="Action")

class Policy(Generic[A]):

    def __init__(self, 
                 network_input: FloatTensor, 
                 network_output: FloatTensor, 
                 actions: Tuple[A,...], 
                 strategy: Callable[["Policy[A]"],A]) -> None:
        super().__init__()

        if int(network_output.nelement()) != len(actions):
            raise ValueError(f"action size of {len(actions)} differs from policy tensor of size {int(network_output.nelement())}")
        
        self._network_input = network_input
        self._network_output = network_output
        self._actions = actions
        self._strategy = strategy

    def action(self) -> A:
        return self._strategy(self)

    def greedy_max(self) -> A:
        return self._actions[int(np.argmax(self.confidences()))]
    
    def greedy_min(self) -> A:
        return self._actions[int(np.argmin(self.confidences()))]
    
    def random(self) -> A:
        return choice(self._actions)

    def epsilon(self, exploration_rate: float, exploit_routine: Callable[["Policy[A]"],A]) -> A:
        if random() < exploration_rate:
            return self.random()
        else:
            return exploit_routine(self)
        
    def epsilon_greedy(self, exploration_rate: float) -> A:
        return self.epsilon(exploration_rate, lambda _: self.greedy_max())

    def weighted_choice(self) -> A:
        confidences = self.confidences()
        if sum(confidences) > 0:
            return choices(self._actions, weights=confidences)[0] 
        else:
            return self.random()

    def gradients(self, action: A) -> NDArray[np.float32]:
        self._network_input.grad = None
        idx = self._actions.index(action)
        self._network_output[idx].backward(retain_graph=True)
        assert self._network_input.grad is not None
        gradients = cast(FloatTensor, self._network_input.grad)
        return gradients.numpy()
    
    def saliency(self, action: A) -> NDArray[np.float32]:
        gradients = self.gradients(action)
        grads_abs: NDArray[np.float32] = np.abs(gradients)
        grad_abs_max = grads_abs.max()
        if grad_abs_max == 0:
            return np.zeros_like(grads_abs)
        else:
            return grads_abs / grads_abs.max()
    
    def action_space(self) -> Tuple[A,...]:
        return self._actions
        
    def confidence_map(self) -> Dict[A,float]:
        return dict(self)
    
    def confidences(self) -> Tuple[float,...]:
        return tuple(self.confidence_map().values())
    
    def features(self) -> NDArray[np.float32]:
        return cast(NDArray[Any], self._network_input\
            .detach()\
            .cpu()\
            .numpy()).astype(np.float32)
    
    def __iter__(self) -> Iterator[Tuple[A,float]]:
        for action,confidence in zip(self._actions, self._network_output):
            yield action,float(confidence.item())