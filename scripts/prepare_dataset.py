import logging
from pathlib import Path
from typing import Optional

import typer

from asr.config import ExperimentConfig
from asr.data import dataset_registry
from asr.utils import init_logging

app = typer.Typer(help="Utilities to download and preprocess Vietnamese ASR datasets.")


@app.command()
def list_datasets() -> None:
    """List registered dataset builders."""

    for name in dataset_registry.keys():
        typer.echo(f"- {name}")


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, help="Path to experiment YAML config"),
    dataset_name: Optional[str] = typer.Option(None, help="Override dataset name"),
    skip_download: bool = typer.Option(False, help="Skip dataset download step"),
) -> None:
    """Download and preprocess dataset as described in the config file."""

    init_logging()
    logger = logging.getLogger('dataset.prepare')

    cfg = ExperimentConfig.from_yaml(config)
    dataset_name = dataset_name or cfg.dataset.name

    builder = dataset_registry.create(
        dataset_name,
        dataset_dir=cfg.dataset.root_dir,
        processed_dir=cfg.dataset.processed_dir,
    )

    logger.info("Preparing dataset '%s'", dataset_name)

    if not skip_download:
        logger.info("Downloading raw dataset to %s", cfg.dataset.root_dir)
        builder.download(**cfg.dataset.download_kwargs)

    processed_dataset = builder.prepare(**cfg.dataset.prepare_kwargs)
    splits = ", ".join(processed_dataset.keys())
    logger.info("Completed preprocessing. Saved splits: %s", splits)


if __name__ == '__main__':
    app()
