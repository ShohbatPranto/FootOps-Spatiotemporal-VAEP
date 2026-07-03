from dataclasses import dataclass

@dataclass
class PitchDimensions:
    """Standardize dimensions for SPADL conversion."""
    SB_LENGTH: float = 120.0
    SB_WIDTH: float = 80.0
    SPADL_LENGTH: float = 105.0
    SPADL_WIDTH: float = 68.0

# Define which StatsBomb events we actually care about for valuation
ACTION_EVENTS = ["Pass", "Carry", "Dribble", "Shot", "Clearance", "Interception", "Foul Won"]