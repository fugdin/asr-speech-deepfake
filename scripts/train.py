import logging
from pathlib import Path
from typing import Optional

import typer

from asr.config import ExperimentConfig
from asr.training import run_experiment
from asr.utils import init_logging

app = typer.Typer(help='Fine-tune Vietnamese ASR models using HuggingFace Transformers.')


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, help='Path to experiment YAML configuration'),
    output_dir: Optional[Path] = typer.Option(None, help='Override output directory for checkpoints/logs'),
) -> None:
    init_logging()
    logger = logging.getLogger('train')

    cfg = ExperimentConfig.from_yaml(config)
    if output_dir is not None:
        cfg.training.output_dir = output_dir
        logger.info('Overriding output_dir to %s', output_dir)

    run_experiment(cfg)


if __name__ == '__main__':
    app()
