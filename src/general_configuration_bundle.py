from dataclasses import dataclass

@dataclass
class GeneralConfigurationBundle:
    method: str | dict
    atoms: str | list
    coordinates: str | list
