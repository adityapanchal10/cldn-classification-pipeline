import shutil


def save_metadata(config_path, run_dir):

    shutil.copy(config_path, run_dir / "config.yaml")
