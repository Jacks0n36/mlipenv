import os
import logging

from mlipenv.exec.util import find_file

logger = logging.getLogger(__name__)

CALCULATOR_ALIASES = {} # maps calculator types back to calculator names

CALCULATOR_REGISTRY = {}
def register_calculator(name, calc_factory=None):
    if calc_factory is None:
        # used as a decorator i.e.
        # @register_calculator(name)
        # def calc(...): ...
        def register(calc_factory):
            return register_calculator(name, calc_factory)
        return register
    else:
        CALCULATOR_REGISTRY[name] = calc_factory
        return calc_factory

def get_calc(*, calculator=None, **calculator_options):
    if calculator is None:
        calculator = os.environ.get("CALCULATOR")
    if isinstance(calculator, str):
        calculator = CALCULATOR_REGISTRY[calculator]
    return calculator(**calculator_options)

MODEL_CACHE_DIR = "MODEL_CACHE_DIR"
DEFAULT_CACHE_DIR = "DEFAULT_MODEL_CACHE_DIR"
def find_module_files(*files):
    cache_locs = []
    if MODEL_CACHE_DIR in os.environ:
        cache_locs.append(os.environ[MODEL_CACHE_DIR])
    if DEFAULT_CACHE_DIR in os.environ:
        cache_locs.append(os.environ[DEFAULT_CACHE_DIR])
    if len(cache_locs) == 0:
        raise ValueError(f"either `{MODEL_CACHE_DIR}` or `{DEFAULT_CACHE_DIR}` must be set at the environment level")
    
    module_files = {file: "" for file in files}
    unfound_files = list(files)
    for cache_dir in cache_locs:
        for file in unfound_files:
            try:
                hit = find_file(file, cache_dir)
                if hit is not None:
                    module_files[file] = hit
                    unfound_files.remove(file)
            except:
                pass
    return module_files

def get_fairchem_predict_unit(model, device):
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit
    from omegaconf import OmegaConf
    model_file = f"{model}.pt" if model[-3:] != ".pt" else model
    atom_refs_file = "iso_atom_elem_refs.yaml"
    module_files = find_module_files(model_file, atom_refs_file)
    model_path = module_files[model_file]
    atom_refs_path = module_files[atom_refs_file]
    atom_refs = OmegaConf.load(atom_refs_path)
    return load_predict_unit(model_path, inference_settings="default", device=device, atom_refs=atom_refs)
  
@register_calculator("fairchem")
def get_fairchem_calc(model="uma-s-1p1", device=None, task_name=None, **kwargs):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    CALCULATOR_ALIASES["fairchem"] = FAIRChemCalculator
    predictor = get_fairchem_predict_unit(device, model)
    return FAIRChemCalculator(predictor, task_name=task_name, **kwargs)

@register_calculator("aimnet")
@register_calculator("aimnet2")
def get_aimnet_calc(model_path="aimnet2", **kwargs):
    from aimnet2calc import AIMNet2ASE
    CALCULATOR_ALIASES["aimnet2"] = AIMNet2ASE
    return AIMNet2ASE(base_calc=model_path, **kwargs)

MACE_CALCULATOR_TYPES=["mace_omol", "mace_off", "mace_mp", "mace_anicc", "mace_polar"]
MACE_CALCULATOR_ALIASES=["omol", "off", "mp", "anicc", "polar"]
@register_calculator("mace")
def get_mace_calc(model_path="mace_omol", mace_calculator="omol", device=None, **kwargs):
    import mace.calculators
    CALCULATOR_ALIASES["mace"] = calc_cls #TODO: make this robust

    if mace_calculator:
        mace_calculator = mace_calculator.lower()
    for calculator_type, calculator_alias in zip(MACE_CALCULATOR_TYPES, MACE_CALCULATOR_ALIASES):
        if not mace_calculator or mace_calculator == calculator_type or mace_calculator == calculator_alias:
            try:
                calc_cls = getattr(mace.calculators, calculator_type)
                CALCULATOR_ALIASES["mace"] = calc_cls #TODO: make this robust
                return calc_cls(model=model_path, device=device, **kwargs)
            except:
                logger.warning(f"could not load using MACE calculator class: {calculator_type}.")

@register_calculator("upet")
def get_upet_calc(checkpoint_path=None, model="pet-mad-s", device=None, **kwargs):
    from upet.calculator import UPETCalculator
    if checkpoint_path is not None:
        try:
            return UPETCalculator(checkpoint_path=checkpoint_path, device=device, **kwargs)
        except:
            logger.warning(f"could not load the model from path {checkpoint_path}.")
    return UPETCalculator(model=model, device=device)

def get_orbmat_model(module, **kwargs):
    import orb_models.forcefield.pretrained
    calc_cls = getattr(orb_models.forcefield.pretrained, module)
    model, atoms_adapter = calc_cls(**kwargs)
    return model, atoms_adapter

@register_calculator("orbmat")
def get_orbmat_calc(module="orbmol-v2", **kwargs):
    from orb_models.forcefield.inference.calculator import ORBCalculator
    model, atoms_adapter = get_orbmat_model(module, **kwargs)
    return ORBCalculator(model=model, atoms_adapter=atoms_adapter)