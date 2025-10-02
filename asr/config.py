from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field, validator


class DatasetSettings(BaseModel):
    name: str = Field(..., description="Registry key for the dataset builder")
    root_dir: Path = Field(Path('data/raw'), description="Root directory for raw datasets")
    processed_dir: Path = Field(Path('data/processed'), description="Directory for cached/preprocessed data")
    download_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Extra kwargs forwarded to dataset download")
    prepare_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Extra kwargs forwarded to dataset preparation")

    @validator('root_dir', 'processed_dir', pre=True)
    def _expand_paths(cls, value: Any) -> Path:
        return Path(value).expanduser().absolute()


class ModelSettings(BaseModel):
    architecture: str = Field('whisper', description="Model family identifier (whisper|wav2vec2|conformer)")
    pretrained_name: str = Field(..., description="Name of the pretrained checkpoint to load from Hugging Face Hub")
    language: str = Field('vi', description="Target language identifier")
    task: str = Field('transcribe', description="Task passed to the processor (transcribe|translate)")
    freeze_encoder: bool = Field(False, description="Freeze encoder layers during fine-tuning to reduce compute")
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing to save memory")


class TrainingSettings(BaseModel):
    output_dir: Path = Field(Path('runs/exp'), description="Directory to store checkpoints and logs")
    num_train_epochs: int = Field(5, ge=1)
    learning_rate: float = Field(1e-5, gt=0)
    warmup_ratio: float = Field(0.1, ge=0, le=1)
    per_device_train_batch_size: int = Field(8, ge=1)
    per_device_eval_batch_size: int = Field(8, ge=1)
    gradient_accumulation_steps: int = Field(1, ge=1)
    fp16: bool = Field(True)
    seed: int = Field(42)

    @validator('output_dir', pre=True)
    def _expand_output(cls, value: Any) -> Path:
        return Path(value).expanduser().absolute()


class EvaluationSettings(BaseModel):
    compute_wer: bool = Field(True)
    decoding_strategy: str = Field('greedy')
    num_beams: int = Field(1, ge=1)


class ExperimentConfig(BaseModel):
    dataset: DatasetSettings
    model: ModelSettings
    training: TrainingSettings
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    @classmethod
    def from_yaml(cls, path: Path | str) -> 'ExperimentConfig':
        import yaml

        with Path(path).expanduser().open('r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream)
        return cls.parse_obj(data)

    def to_yaml(self, path: Path | str) -> None:
        import yaml

        with Path(path).expanduser().open('w', encoding='utf-8') as stream:
            yaml.safe_dump(self.dict(), stream, sort_keys=False)


__all__ = ['ExperimentConfig', 'DatasetSettings', 'ModelSettings', 'TrainingSettings', 'EvaluationSettings']
