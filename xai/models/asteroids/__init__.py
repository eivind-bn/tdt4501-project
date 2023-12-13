from typing import *

from ...agents.asteroids import AsteroidsGenome

from .salinet_bot import SalientBot
from .spinner_bot import SpinnerBot
from .v3_bot import V3Bot
from .v4_bot import V4Bot
from .v5_bot import V5Bot
from .v6_bot import V6Bot
from .v7_bot import V7Bot

if TYPE_CHECKING:
    from ...agents.asteroids import AsteroidsPolicy
    from ...games.asteroids import AsteroidsObservation, AsteroidsAction, AsteroidsReward