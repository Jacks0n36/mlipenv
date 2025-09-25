import os
import abc

import numpy as np
from ase import Atoms

from src.calculators import get_calc

class BaseOptimizationRunner:
    def __init__(self, atoms, coordinates):
        self.atoms = atoms
        self.coordinates = coordinates

    def export_results(self, output_dir):
        atom_symbols, coordinates, gradients = self.format_results()
        np.savez(os.path.join(output_dir, "atoms.npz"), atom_symbols)
        np.savez(os.path.join(output_dir, "coordinates.npz"), coordinates)
        np.savez(os.path.join(output_dir, "gradients.npz"), gradients)

    def format_results(self):
        results = self.results
        atom_symbols = np.array([self.get_atom_symbols(obj) for obj in results])
        coordinates = np.array([self.get_coordinates(obj) for obj in results])
        gradients = np.array([self.get_gradients(obj) for obj in results])
        return atom_symbols, coordinates, gradients
        
    @abc.abstractmethod
    def run(self):
        ...

    @abc.abstractmethod
    def get_atom_symbols(self, obj):
        ...
    @abc.abstractmethod
    def get_coordinates(self, obj):
        ...
    @abc.abstractmethod
    def get_gradients(self, obj):
        ...
    @abc.abstractmethod
    def get_single_point_energy(self, obj):
        ...
    # option to write .trj and .log files to a specified place


class ASEOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, atoms, coordinates, config):
        super().__init__(atoms, coordinates)
        self.load_config_with_defaults(config)

    def load_config_with_defaults(self, config):
        from src.optimization_options import ASEOptOpts
        self.options = ASEOptOpts(**config)
    
    def run(self):
        optimized_atoms = [self.run_opt(a, c) for a, c in zip(self.atoms, self.coordinates)]
        self.results = optimized_atoms
    
    def get_atom_symbols(self, obj):
        return np.array(obj.get_chemical_symbols())
    def get_coordinates(self, obj):
        return np.array(obj.get_positions())
    def get_gradients(self, obj):
        return np.array(obj.calc.get_forces())
    def get_single_point_energy(self, obj):
        return np.array(obj.calc.get_potential_energy())

    def run_opt(self, atom_symbols, coordinates):
        atoms = Atoms(symbols=atom_symbols, positions=coordinates)
        atoms.info["charge"] = self.options.charge
        atoms.info["spin"] = self.options.spin
        calc = get_calc()
        atoms.calc = calc
        atoms = self._run_opt(atoms)
        return atoms
    
    def _run_opt(self, atoms):
        from ase.optimize import BFGS
        opt = BFGS(atoms)
        opt.run(fmax=self.options.fmax, steps=self.options.steps)
        return atoms


class SciPyOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, atoms, coordinates, config):
        super().__init__(atoms, coordinates)
    
    def run(self):
        ...


class MarksOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, atoms, coordinates, config):
        super().__init__(atoms, coordinates)
    
    def run(self):
        ...
