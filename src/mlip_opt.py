import os
import json

from src.managers import OptimizationManager, EnergyManager

# config:
#   method: { type: , options: {}}
#   atoms
#   coordinates

# method: { 
#   type: optimize, 
#   options: {
#       optimizer: {
#           type:
#           options: {}
#           }
#       }

# method: { 
#   type: energy, 
#   options: {
#       order:
#       }

def load_config(config_bundle):
    if isinstance(config_bundle, str):
        if os.path.exists(config_bundle):
            with open(config_bundle, "r") as f:
                config = json.load(f)
        else:
            try:
                config = json.loads(config_bundle)
            except:
                raise NotImplementedError("Cannot load from string that is neither a valid path to nor formatted JSON itself.")
    
    elif isinstance(config_bundle, dict):
        config = config_bundle
    else:
        raise NotImplementedError(f"Intractable input type: {type(config_bundle)}")
    return config

method_info_key = "method"
method_key = "type"
def get_runner_for_method(config):
    method_info = config[method_info_key]
    method = method_info[method_key]
    if method == "optimize":
        return OptimizationManager(config)
    elif method == "energy":
        return EnergyManager(config)
    else:
        raise NotImplementedError(f"Unknown method type: {method}")

def call_to_mlip_server(config_bundle):
    config = load_config(config_bundle)
    runner = get_runner_for_method(config)
    runner.run()


call_to_mlip_server("test/optimize.json")