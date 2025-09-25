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

# this is ugly!
DEFAULT_CHARGE = 0
DEFAULT_SPIN = 1
DEFAULT_FMAX = 0.02
DEFAULT_OUT_PATH = "./"
DEFAULT_OUT_FILE = "opt.npz"
DEFAULT_RETURN = "all"
DEFAULT_MAX_STEPS = 20
DEFAULT_CONFIG = {
    "charge": DEFAULT_CHARGE,
    "spin": DEFAULT_SPIN,
    "fmax": DEFAULT_FMAX,
    "steps": DEFAULT_MAX_STEPS,
    "out_path": DEFAULT_OUT_PATH,
    "out_file": DEFAULT_OUT_FILE,
    "return_info": DEFAULT_RETURN,
}

class ASEOptimizationRunner(BaseOptimizationRunner):
    def __init__(self, atoms, coordinates, config):
        super().__init__(atoms, coordinates)
        self.load_config_with_defaults(config)

    def load_config_with_defaults(self, config):
        self.charge = config["charge"] if "charge" in config else DEFAULT_CONFIG["charge"]
        self.spin = config["spin"] if "spin" in config else DEFAULT_CONFIG["spin"]
        self.fmax = config["fmax"] if "fmax" in config else DEFAULT_CONFIG["fmax"]
        self.steps = config["steps"] if "steps" in config else DEFAULT_CONFIG["steps"]
    
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
        atoms.info["charge"] = self.charge
        atoms.info["spin"] = self.spin
        calc = get_calc()
        atoms = self._run_opt(atoms, calc)
        return atoms
    
    def _run_opt(self, atoms, calc):
        fmax = self.fmax
        steps = self.steps
        from ase.optimize import BFGS
        atoms.calc = calc
        opt = BFGS(atoms)
        opt.run(fmax=fmax, steps=steps)
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
