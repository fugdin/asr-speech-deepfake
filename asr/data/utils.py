from __future__ import annotations

import re
import unicodedata
from typing import Dict

from datasets import Audio, Dataset, DatasetDict

PUNCTUATION_REGEX = re.compile(r"[\.,\?!:\-\;\(\)\[\]\{\}\"\']")


def normalize_vietnamese_text(text: str) -> str:
    if not text:
        return text
    text = text.strip().lower()
    text = PUNCTUATION_REGEX.sub(" ", text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cast_audio_column(dataset: Dataset | DatasetDict, column: str = 'audio', sampling_rate: int = 16000) -> Dataset | DatasetDict:
    if isinstance(dataset, DatasetDict):
        return DatasetDict({split: cast_audio_column(ds, column=column, sampling_rate=sampling_rate) for split, ds in dataset.items()})
    audio_feature = Audio(sampling_rate=sampling_rate)
    return dataset.cast_column(column, audio_feature)


def map_transcript_column(
    dataset: Dataset | DatasetDict,
    text_column: str = 'sentence',
    target_column: str = 'transcript',
    normalizer = normalize_vietnamese_text,
    num_proc: int = 1,
) -> Dataset | DatasetDict:
    def _map_fn(batch: Dict) -> Dict:
        text = batch[text_column]
        if isinstance(text, str):
            normalized = normalizer(text) if normalizer else text
        else:
            normalized = text
        return {target_column: normalized}

    if isinstance(dataset, DatasetDict):
        return DatasetDict({split: map_transcript_column(ds, text_column, target_column, normalizer, num_proc) for split, ds in dataset.items()})

    mapped = dataset.map(_map_fn, num_proc=num_proc, desc=f'Normalizing transcripts -> {target_column}')
    return mapped


__all__ = ['normalize_vietnamese_text', 'cast_audio_column', 'map_transcript_column']
