from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from transformers import pipeline

from asr.config import ExperimentConfig
from asr.models import load_model_and_processor
from asr.utils import resample_audio


def _resolve_device(device: Optional[str] = None) -> Tuple[str, int]:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = device.lower()
    if device.startswith('cuda') and torch.cuda.is_available():
        if ':' in device:
            index = int(device.split(':')[1])
        else:
            index = 0
        return device, index
    return 'cpu', -1


def build_asr_pipeline(
    architecture: str,
    pretrained_name: str,
    device: Optional[str] = None,
) -> Tuple[object, Dict[str, str]]:
    """Return a HuggingFace ASR pipeline and metadata for downstream usage."""

    resolved_device, pipeline_device = _resolve_device(device)
    model, processor = load_model_and_processor(
        architecture=architecture,
        pretrained_name=pretrained_name,
        device=resolved_device,
    )

    feature_extractor = getattr(processor, 'feature_extractor', None)
    tokenizer = getattr(processor, 'tokenizer', None)

    asr_pipeline = pipeline(
        task='automatic-speech-recognition',
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=pipeline_device,
    )

    metadata = {
        'architecture': architecture,
        'pretrained_name': pretrained_name,
        'device': resolved_device,
        'sampling_rate': str(getattr(feature_extractor, 'sampling_rate', 16000)),
    }
    return asr_pipeline, metadata


def transcribe_file(
    audio_path: Path | str,
    config_path: Optional[Path | str] = None,
    architecture: Optional[str] = None,
    pretrained_name: Optional[str] = None,
    device: Optional[str] = None,
    chunk_length_s: float = 30.0,
    return_timestamps: bool = False,
) -> Dict:
    """Transcribe a single audio file using either a config or manual arguments."""

    if config_path:
        cfg = ExperimentConfig.from_yaml(config_path)
        architecture = cfg.model.architecture
        pretrained_name = cfg.model.pretrained_name
        language = cfg.model.language
        task = cfg.model.task
    else:
        language = 'vi'
        task = 'transcribe'

    if architecture is None or pretrained_name is None:
        raise ValueError('Provide either a config file or both architecture and pretrained_name.')

    asr_pipeline, metadata = build_asr_pipeline(architecture, pretrained_name, device=device)

    call_kwargs: Dict = {}
    if chunk_length_s:
        call_kwargs['chunk_length_s'] = chunk_length_s

    if architecture.lower() == 'whisper':
        call_kwargs['generate_kwargs'] = {'language': language, 'task': task}
        if return_timestamps:
            call_kwargs['return_timestamps'] = True

    result = asr_pipeline(str(audio_path), **call_kwargs)

    if isinstance(result, list):
        text = ' '.join(chunk['text'].strip() for chunk in result)
        payload = {'text': text}
        if return_timestamps:
            payload['chunks'] = result
        payload['metadata'] = metadata
        return payload

    payload = {'text': result.get('text', '').strip(), 'metadata': metadata}
    if return_timestamps and 'chunks' in result:
        payload['chunks'] = result['chunks']
    return payload


def transcribe_waveform(
    waveform: np.ndarray,
    sampling_rate: int,
    architecture: str,
    pretrained_name: str,
    device: Optional[str] = None,
) -> str:
    pipeline_instance, metadata = build_asr_pipeline(architecture, pretrained_name, device=device)
    target_sr = int(metadata.get('sampling_rate', 16000))
    if sampling_rate != target_sr:
        waveform = resample_audio(waveform, sampling_rate, target_sr)
        sampling_rate = target_sr
    output = pipeline_instance({'array': waveform, 'sampling_rate': sampling_rate})
    return output.get('text', '').strip()


__all__ = ['build_asr_pipeline', 'transcribe_file', 'transcribe_waveform']
