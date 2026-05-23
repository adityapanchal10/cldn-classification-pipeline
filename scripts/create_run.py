from pathlib import Path
from datetime import datetime


def create_run(cfg):

    run_name = cfg["experiment"]["name"]

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        Path(cfg["paths"]["runs_root"])
        / f"{run_name}_{timestamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return run_dir