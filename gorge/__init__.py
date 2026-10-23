"""gorge: eat entire drives at once by phasing data directly into model weights."""

__version__ = "0.1.0"

from .phasing import (  # noqa: F401
    Phaser,
    PhasedWeights,
    dim_for_capacity,
    embed,
    four_grams,
    load_weights,
    parse_capacity,
    save_weights,
    shannon_entropy,
)
from .types import MAJORS, MAJOR_ID, detect_type  # noqa: F401