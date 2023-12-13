from typing import *

from ..action import Action

class AsteroidsAction(Action):
    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    UP_RIGHT = 6
    UP_LEFT = 7
    UP_FIRE = 8
    RIGHT_FIRE = 9
    LEFT_FIRE = 10
    DOWN_FIRE = 11
    UP_RIGHT_FIRE = 12
    UP_LEFT_FIRE = 13