from __future__ import annotations

from pathlib import Path
from typing import Dict

from datasets import DatasetDict, load_dataset

from .dataset_config import BaseDatasetBuilder
from .registry import dataset_registry, DatasetConfig
from .utils import cast_audio_column, map_transcript_column, normalize_vietnamese_text


class CustomDatasetBuilder(BaseDatasetBuilder):
    config_name = 'custom_manifest'

    def download(self, **kwargs) -> None:  # pragma: no cover - user provided data
        raise RuntimeError('Custom dataset builder does not support automated download; provide local data.')

    def prepare(
        self,
        manifests: Dict[str, Path],
        audio_column: str = 'audio_filepath',
        text_column: str = 'text',
        speaker_column: str | None = 'speaker_id',
        target_sample_rate: int = 16000,
        normalize_transcripts: bool = True,
        num_workers: int = 4,
        **kwargs,
    ) -> DatasetDict:
        if not manifests:
            raise ValueError('Provide at least one manifest file path.')

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        data_files = {split: str(Path(path).expanduser()) for split, path in manifests.items()}
        dataset = load_dataset('json', data_files=data_files, field=None)

        dataset = cast_audio_column(dataset, column=audio_column, sampling_rate=target_sample_rate)
        if audio_column != 'audio':
            dataset = dataset.rename_column(audio_column, 'audio')

        if normalize_transcripts:
            dataset = map_transcript_column(
                dataset,
                text_column=text_column,
                target_column='transcript',
                normalizer=normalize_vietnamese_text,
                num_proc=num_workers,
            )
        else:
            dataset = dataset.rename_column(text_column, 'transcript')

        if speaker_column:
            try:
                dataset = dataset.rename_column(speaker_column, 'speaker_id')
            except ValueError:
                # Column might be missing in some splits; ignore gracefully.
                pass

        for split in dataset.keys():
            keep = {'audio', 'transcript'}
            if 'speaker_id' in dataset[split].column_names:
                keep.add('speaker_id')
            drop_cols = [col for col in dataset[split].column_names if col not in keep]
            if drop_cols:
                dataset[split] = dataset[split].remove_columns(drop_cols)

        dataset.save_to_disk(self.processed_dir.as_posix())
        return dataset

    def metadata(self) -> Dict[str, str]:
        return {
            'language': 'vi',
            'speakers': 'User supplied',
            'license': 'User supplied',
        }


dataset_registry.register(
    DatasetConfig(
        name='custom',
        builder=CustomDatasetBuilder,
        description='User-provided JSON manifests with audio paths and transcripts',
    )
)


__all__ = ['CustomDatasetBuilder']
