import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
import subprocess
import sys

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
    "test_regression_model"
]

# This automatically reads in the configuration
@hydra.main(version_base=None, config_name='config', config_path='.')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Dynamic artifact names from config
    etl_cfg = config.get("etl", {})
    dc_cfg = config.get("data_check", {})
    mod_cfg = config.get("modeling", {})

    clean_artifact_name = etl_cfg.get("output_artifact", "clean_data.csv")
    clean_artifact_ref = f"{clean_artifact_name}:latest"

    # Shared environment for all subprocesses
    sub_env = os.environ.copy()
    sub_env["WANDB_MODE"] = "online"
    sub_env["PYTHONPATH"] = f"/workspace/src:/workspace:/workspace/venv/lib/python3.10/site-packages:{sub_env.get('PYTHONPATH', '')}"
    py_bin = "/workspace/venv/bin/python"

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            cmd = [
                py_bin,
                os.path.join("components", "get_data", "run.py"),
                str(etl_cfg.get("sample", "sample1.csv")),
                "sample.csv",
                "raw_data",
                "Raw file as downloaded",
            ]
            subprocess.run(cmd, check=True, env=sub_env)

        if "basic_cleaning" in active_steps:
                       
            cmd = [
                py_bin,
                os.path.join("src", "basic_cleaning", "run.py"),
                "--input_artifact", "sample.csv:latest",
                "--output_artifact", clean_artifact_name,
                "--output_type", "dataset",
                "--output_description", "Cleaned data of Airbnb dataset",
                "--min_price", str(etl_cfg.get("min_price", 10)),
                "--max_price", str(etl_cfg.get("max_price", 350)),
            ]
            subprocess.run(cmd, check=True, env=sub_env)
                      
        if "data_check" in active_steps:
            kl_thresh = dc_cfg.get("kl_threshold", 0.2)
            min_p = etl_cfg.get("min_price", 10)
            max_p = etl_cfg.get("max_price", 350)
            cmd = [
                py_bin,
                "-m",
                "pytest",
                "src/data_check",
                "-vv",
                "--csv", clean_artifact_ref,
                "--ref", "sample.csv:latest",
                "--kl_threshold", str(kl_thresh),
                "--min_price", str(min_p),
                "--max_price", str(max_p),
            ]
            subprocess.run(cmd, check=True, env=sub_env)

        if "data_split" in active_steps:
            cmd = [
            py_bin,
                os.path.join("components", "train_val_test_split", "run.py"),
                clean_artifact_ref,              
                str(mod_cfg.get("test_size", 0.2)),
                "--random_seed", str(mod_cfg.get("random_seed", 42)),
                "--stratify_by", str(mod_cfg.get("stratify_by", "none")),
            ]
            subprocess.run(cmd, check=True, env=sub_env)
            
        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
            rf_params = dict(mod_cfg.get("random_forest", {}).items())
            with open(rf_config, "w+") as fp:
                json.dump(rf_params, fp)
            
            cmd = [
                py_bin,
                os.path.join("components", "train_random_forest", "run.py"),
                "trainval_data.csv:latest",
                str(mod_cfg.get("val_size", 0.2)),
                "--random_seed", str(mod_cfg.get("random_seed", 42)),
                "--stratify_by", str(mod_cfg.get("stratify_by", "none")),
                "--rf_config", rf_config,
                "--max_tfidf_features", str(mod_cfg.get("max_tfidf_features", 5)),
                "--output_artifact", "random_forest_export",
            ]
            subprocess.run(cmd, check=True, env=sub_env)

        if "test_regression_model" in active_steps and "prod" in steps_par:
            cmd = [
                py_bin,
                os.path.join("components", "test_regression_model", "run.py"),
                "--mlflow_model", "random_forest_export:prod",
                "--test_dataset", "test_data.csv:latest",
            ]
            subprocess.run(cmd, check=True, env=sub_env)

if __name__ == "__main__":
    go()
