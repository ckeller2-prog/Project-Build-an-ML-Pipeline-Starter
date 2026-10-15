#!/usr/bin/env python
import argparse
import logging
import pathlib
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact...")
    artifact_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_path)

    logger.info("Dropping outliers by price...")
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Dropping outliers by price and invalid coordinates...")
    idx = (
        df["price"].between(args.min_price, args.max_price)
        & df["longitude"].between(-74.25, -73.50)
        & df["latitude"].between(40.5, 41.2)
    )
    df = df[idx].copy()

    # Save the cleaned dataset locally
    output_filename = "clean_sample.csv"
    df.to_csv(output_filename, index=False)

    logger.info("Uploading cleaned dataset to Weights & Biases...")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_filename)
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic data cleaning step")

    parser.add_argument("--input_artifact", type=str, required=True, help="Input artifact from W&B")
    parser.add_argument("--output_artifact", type=str, required=True, help="Cleaned output artifact name")
    parser.add_argument("--output_type", type=str, required=True, help="Type of the output artifact")
    parser.add_argument("--output_description", type=str, required=True, help="Description of the output artifact")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum price to consider")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum price to consider")

    args = parser.parse_args()
    go(args)
