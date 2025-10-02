from __future__ import annotations

from typing import Dict

from datasets import DatasetDict, load_dataset

from .dataset_config import BaseDatasetBuilder
from .registry import dataset_registry, DatasetConfig
from .utils import cast_audio_column, map_transcript_column, normalize_vietnamese_text


class CommonVoiceDatasetBuilder(BaseDatasetBuilder):
    config_name = 'common_voice_vi'

    def download(self, version: str = '13_0', use_auth_token: str | None = None, **kwargs) -> None:
        load_dataset(
            f'mozilla-foundation/common_voice_{version}',
            'vi',
            cache_dir=self.dataset_dir.as_posix(),
            use_auth_token=use_auth_token,
        )

    def prepare(
        self,
        version: str = '13_0',
        use_auth_token: str | None = None,
        target_sample_rate: int = 16000,
        normalize_transcripts: bool = True,
        include_test_split: bool = True,
        num_workers: int = 4,
        **kwargs,
    ) -> DatasetDict:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset(
            f'mozilla-foundation/common_voice_{version}',
            'vi',
            cache_dir=self.dataset_dir.as_posix(),
            use_auth_token=use_auth_token,
        )

        keep_splits = {'train', 'validation'}
        if include_test_split and 'test' in dataset:
            keep_splits.add('test')
        dataset = DatasetDict({split: dataset[split] for split in dataset.keys() if split in keep_splits})

        dataset = dataset.filter(lambda row: bool(row.get('sentence')), num_proc=num_workers)
        dataset = cast_audio_column(dataset, column='audio', sampling_rate=target_sample_rate)

        if normalize_transcripts:
            dataset = map_transcript_column(
                dataset,
                text_column='sentence',
                target_column='transcript',
                normalizer=normalize_vietnamese_text,
                num_proc=num_workers,
            )
        else:
            dataset = dataset.rename_column('sentence', 'transcript')

        for split in dataset.keys():
            reserved_columns = {'audio', 'transcript', 'client_id', 'up_votes', 'down_votes'}
            drop_cols = [col for col in dataset[split].column_names if col not in reserved_columns]
            if drop_cols:
                dataset[split] = dataset[split].remove_columns(drop_cols)

        dataset.save_to_disk(self.processed_dir.as_posix())
        return dataset

    def metadata(self) -> Dict[str, str]:
        return {
            'language': 'vi',
            'speakers': 'Crowd-sourced Vietnamese voices (Common Voice)',
            'license': 'CC-0',
        }


dataset_registry.register(
    DatasetConfig(
        name='common_voice_vi',
        builder=CommonVoiceDatasetBuilder,
        description='Mozilla Common Voice Vietnamese subset',
    )
)


__all__ = ['CommonVoiceDatasetBuilder']
