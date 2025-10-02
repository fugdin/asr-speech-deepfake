from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional


@dataclass
class DatasetConfig:
    """Lightweight configuration object used by dataset builder registry."""

    name: str
    builder: Callable[..., 'BaseDatasetBuilder']
    description: str


class BaseDatasetBuilder:
    """Abstract base class for dataset builders."""

    config_name: str = "base"

    def __init__(self, dataset_dir: Path, processed_dir: Path, **kwargs) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().absolute()
        self.processed_dir = Path(processed_dir).expanduser().absolute()
        self.kwargs = kwargs

    def download(self, **kwargs) -> None:
        raise NotImplementedError

    def prepare(self, num_workers: int = 4, **kwargs) -> Path:
        raise NotImplementedError

    def metadata(self) -> Dict[str, str]:
        return {}

    def available_speakers(self) -> Optional[Iterable[str]]:
        return None


__all__ = ['DatasetConfig', 'BaseDatasetBuilder']
