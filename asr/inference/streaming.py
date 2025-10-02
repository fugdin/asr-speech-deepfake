from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from asr.utils import resample_audio


@dataclass
class StreamingTranscriber:
    """Maintain a rolling audio buffer and emit transcripts per chunk."""

    pipeline: object
    sample_rate: int
    chunk_length_s: float = 5.0
    stride_s: float = 1.0
    generate_kwargs: Optional[Dict] = None

    _buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def __post_init__(self) -> None:
        if self.chunk_length_s <= 0:
            raise ValueError('chunk_length_s must be positive')
        if self.stride_s <= 0:
            raise ValueError('stride_s must be positive')
        if self.stride_s > self.chunk_length_s:
            raise ValueError('stride_s must be <= chunk_length_s')

        self.chunk_samples = int(self.chunk_length_s * self.sample_rate)
        self.stride_samples = int(self.stride_s * self.sample_rate)
        if self.generate_kwargs is None:
            self.generate_kwargs = {}

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)

    def _infer(self, samples: np.ndarray) -> str:
        payload = {
            'array': samples.astype(np.float32),
            'sampling_rate': self.sample_rate,
        }
        result = self.pipeline(payload, generate_kwargs=self.generate_kwargs)
        if isinstance(result, dict):
            return result.get('text', '').strip()
        if isinstance(result, list):
            return ' '.join(chunk.get('text', '') for chunk in result).strip()
        return ''

    def append(self, audio_chunk: np.ndarray, sampling_rate: int) -> List[str]:
        if audio_chunk.size == 0:
            return []

        if sampling_rate != self.sample_rate:
            audio_chunk = resample_audio(audio_chunk, sampling_rate, self.sample_rate)

        audio_chunk = audio_chunk.astype(np.float32)
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=0)

        self._buffer = np.concatenate([self._buffer, audio_chunk])

        transcripts: List[str] = []
        while self._buffer.size >= self.chunk_samples:
            window = self._buffer[: self.chunk_samples]
            transcript = self._infer(window)
            if transcript:
                transcripts.append(transcript)
            self._buffer = self._buffer[self.stride_samples :]
        return transcripts

    def flush(self) -> Optional[str]:
        if self._buffer.size == 0:
            return None
        transcript = self._infer(self._buffer)
        self.reset()
        return transcript or None


__all__ = ['StreamingTranscriber']
