import yaml

from pathlib import Path

from scripts.create_run import create_run
from scripts.save_metadata import save_metadata
from scripts.execute_notebook import execute_notebook
from scripts.set_seed import set_seed

import argparse

# Allow passing CONFIG_PATH as a positional CLI arg; default unchanged
parser = argparse.ArgumentParser(description="Launch experiment from config YAML")
parser.add_argument("config", help="Path to YAML config file")
args = parser.parse_args()
CONFIG_PATH = args.config

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)


run_dir = create_run(cfg)

save_metadata(CONFIG_PATH, run_dir)


# -------------------
# TRAIN
# -------------------

if cfg["evaluation"]["mode"] == "grouped":
    execute_notebook(
        input_path="notebooks/train_grouped_holdout.ipynb",

        output_path=(
            run_dir / "train.ipynb"
        ),

        parameters={
            "config_path": CONFIG_PATH,
            "run_dir": str(run_dir)
        }
    )
elif cfg["evaluation"]["mode"] == "single":
    execute_notebook(
        input_path="notebooks/train_single_split.ipynb",

        output_path=(
            run_dir / "train.ipynb"
        ),

        parameters={
            "config_path": CONFIG_PATH,
            "run_dir": str(run_dir)
        }
    )

# -------------------
# EVALUATE
# -------------------

execute_notebook(
    input_path="notebooks/evaluate.ipynb",

    output_path=(
        run_dir / "evaluate.ipynb"
    ),

    parameters={
        "config_path": CONFIG_PATH,
        "run_dir": str(run_dir)
    }
)


print(f"Experiment complete: {run_dir}")