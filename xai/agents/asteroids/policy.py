from typing import *

from ...agents import Policy

if TYPE_CHECKING:
    from ...games.asteroids import AsteroidsAction

class AsteroidsPolicy(Policy["AsteroidsAction"]):
    pass