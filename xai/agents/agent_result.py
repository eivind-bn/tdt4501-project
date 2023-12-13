from typing import *

if TYPE_CHECKING:
    from ..agents import Fitness

class AgentResult(TypedDict):
    steps_played: int
    game_reward: int|float
    fitness: "Fitness"
