import os
import abc

import numpy as np

from src.runners import ASEOptimizationRunner, SciPyOptimizationRunner, MarksOptimizationRunner

atoms_key = "atoms"
coordinates_key = "coordinates"
output_dir_key = "output_dir"
class BaseManager:
    def __init__(self, config):
        self.atoms = self.load_atoms(config[atoms_key])
        self.coordinates = self.load_coordinates(config[coordinates_key])
        self.output_dir = config[output_dir_key]

    def _load_parameter(self, parameter_bundle):
        if isinstance(parameter_bundle, str):
            if os.path.exists(parameter_bundle):
                data = np.load(parameter_bundle)
                parameter = [data[key] for key in data.files]
        elif isinstance(parameter_bundle, list):
            parameter = parameter_bundle
        return parameter

    def load_coordinates(self, coordinates_bundle):
        try:
            coordinates = self._load_parameter(coordinates_bundle)
        except:
            raise NotImplementedError(f"Could not load coordinates from: {coordinates_bundle}")
        return coordinates
    
    def load_atoms(self, atoms_bundle):
        try:
            coordinates = self._load_parameter(atoms_bundle)
        except:
            raise NotImplementedError(f"Could not load atoms from: {atoms_bundle}")
        return coordinates
        
    @abc.abstractmethod
    def run(self):
        ...


class EnergyManager(BaseManager):
    def __init__(self, config):
        super().__init__(config)
    
    def run(self):
        self.compute_energy()
    
    def compute_energy(self):
        ...


method_info_key = "method"
method_key = "type"
method_options_key = "options"
optimizer_info_key = "optimizer"
optimizer_key = "type"
optimizer_options_key = "options"
class OptimizationManager(BaseManager):
    def __init__(self, config):
        super().__init__(config)
        self.options = config[method_info_key][method_options_key]

    def get_optimization_scheme(self):
        requested_optimizer = self.options[optimizer_info_key][optimizer_key]
        optimizer_options = self.options[optimizer_info_key][optimizer_options_key]
        if requested_optimizer == "ase":
            return ASEOptimizationRunner(self.atoms, self.coordinates, optimizer_options)
        elif requested_optimizer == "scipy":
            return SciPyOptimizationRunner(self.atoms, self.coordinates, optimizer_options)
        elif "mark" in requested_optimizer.lower():
            return MarksOptimizationRunner(self.atoms, self.coordinates, optimizer_options)
        else:
            raise NotImplementedError(f"Unknown optimizer: {requested_optimizer}")
    
    def run(self):
        optimization_runner = self.get_optimization_scheme()
        optimization_runner.run()
        optimization_runner.export_results(self.output_dir)