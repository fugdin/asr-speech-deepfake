from __future__ import annotations

from typing import Dict

from .dataset_config import BaseDatasetBuilder, DatasetConfig


class DatasetRegistry:
    def __init__(self) -> None:
        self._builders: Dict[str, DatasetConfig] = {}

    def register(self, config: DatasetConfig) -> None:
        if config.name in self._builders:
            raise ValueError(f"Dataset '{config.name}' already registered")
        self._builders[config.name] = config

    def create(self, name: str, *args, **kwargs) -> BaseDatasetBuilder:
        if name not in self._builders:
            raise KeyError(f"Dataset '{name}' is not registered")
        builder_cls = self._builders[name].builder
        return builder_cls(*args, **kwargs)

    def describe(self, name: str) -> str:
        if name not in self._builders:
            raise KeyError(f"Dataset '{name}' is not registered")
        return self._builders[name].description

    def keys(self) -> Dict[str, DatasetConfig].keys:
        return self._builders.keys()


dataset_registry = DatasetRegistry()


__all__ = ['dataset_registry', 'DatasetRegistry']
