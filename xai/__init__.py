from typing import *
from enum import Enum

from .agents import *
from .games import *
from .util import *

from .models.asteroids import SalientBot, SpinnerBot, V3Bot, V4Bot, V5Bot, V6Bot, V7Bot

G = TypeVar("G", bound=Genome)

def load_model(path: str, type: Type[G]) -> G:
    agent = type.load(path)
    if isinstance(agent, type):
        return agent
    else:
        raise TypeError(f"Agent of type: {agent.__class__} is not instance of the requested class: {type}")
