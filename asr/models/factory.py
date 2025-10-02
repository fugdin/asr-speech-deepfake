from __future__ import annotations

from typing import Tuple

from transformers import AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor


def load_model_and_processor(architecture: str, pretrained_name: str, device: str = 'cuda') -> Tuple[object, AutoProcessor]:
    """Factory that loads ASR models based on the requested architecture."""

    architecture = architecture.lower()
    if architecture in {'wav2vec2', 'hubert', 'ctc'}:
        model = AutoModelForCTC.from_pretrained(pretrained_name)
    elif architecture in {'whisper', 'seq2seq'}:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_name)
    else:
        raise ValueError(f'Unknown architecture: {architecture}')

    processor = AutoProcessor.from_pretrained(pretrained_name)

    if device == 'cuda':
        model = model.to('cuda')
    elif device == 'cpu':
        model = model.to('cpu')
    else:
        model = model.to(device)

    return model, processor


__all__ = ['load_model_and_processor']
