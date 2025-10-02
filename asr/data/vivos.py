from __future__ import annotations

from pathlib import Path
from typing import Dict

from datasets import DatasetDict, load_dataset

from .dataset_config import BaseDatasetBuilder
from .registry import dataset_registry, DatasetConfig
from .utils import cast_audio_column, map_transcript_column


class VIVOSDatasetBuilder(BaseDatasetBuilder):
    config_name = 'vivos'

    def download(self, **kwargs) -> None:
        load_dataset('vivos', cache_dir=self.dataset_dir.as_posix(), **kwargs)

    def prepare(
        self,
        num_workers: int = 4,
        target_sample_rate: int = 16000,
        normalize_transcripts: bool = True,
        train_split: str = 'train',
        eval_split: str = 'test',
        **kwargs,
    ) -> DatasetDict:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset(
            'vivos',
            cache_dir=self.dataset_dir.as_posix(), **kwargs,
        )

        dataset = DatasetDict({k: v for k, v in dataset.items() if k in {train_split, eval_split}})
        dataset = cast_audio_column(dataset, column='audio', sampling_rate=target_sample_rate)

        if normalize_transcripts:
            dataset = map_transcript_column(dataset, text_column='sentence', target_column='transcript', num_proc=num_workers)
        else:
            dataset = dataset.rename_column('sentence', 'transcript')

        for split in dataset.keys():
            columns_to_drop = [col for col in dataset[split].column_names if col not in {'audio', 'transcript', 'speaker_id'}]
            if columns_to_drop:
                dataset[split] = dataset[split].remove_columns(columns_to_drop)

        dataset.save_to_disk(self.processed_dir.as_posix())
        return dataset

    def metadata(self) -> Dict[str, str]:
        return {
            'language': 'vi',
            'speakers': '26 speakers (15 male, 11 female) recorded in studio quality',
            'license': 'AIVIVN release',
        }


dataset_registry.register(
    DatasetConfig(
        name='vivos',
        builder=VIVOSDatasetBuilder,
        description='Vietnamese VIVOS corpus (AIVIVN)',
    )
)


__all__ = ['VIVOSDatasetBuilder']
