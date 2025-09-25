def get_calc():
    return get_fairchem_calc()

def get_fairchem_calc():
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    predictor = get_predict_unit("uma-s-1p1", device="cpu")
    calc = FAIRChemCalculator(predictor, task_name="omol")
    return calc

def get_mace_calc():
    ...