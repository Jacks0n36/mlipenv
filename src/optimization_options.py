from dataclasses import dataclass

@dataclass
class ASEOptOpts:
    charge: float = 0.0
    spin: float = 1.0
    fmax: float = 0.02
    steps: int = 20
    logging: str = ""
    
