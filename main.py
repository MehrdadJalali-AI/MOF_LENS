
import yaml
from experiments import run_full_experiments

with open("config.yaml") as f:
    config = yaml.safe_load(f)

run_full_experiments(config)
