from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
from jiwer import wer
from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    set_seed,
)

from asr.config import ExperimentConfig
from asr.models import load_model_and_processor
from asr.utils import init_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class DataCollatorCTCWithPadding:
    """Minimal CTC collator compatible with recent Transformers releases."""

    processor: AutoProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        audio_inputs = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(audio_inputs, padding=self.padding, return_tensors="pt")

        with self.processor.as_target_processor():
            label_inputs = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.pad(label_inputs, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"]
        attention_mask = labels_batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask.ne(1), -100)
        else:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                labels = labels.masked_fill(labels == pad_token_id, -100)

        batch["labels"] = labels
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Lightweight Whisper data collator independent of Transformers internals."""

    processor: AutoProcessor
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"]
        pad_token_id = self.processor.tokenizer.pad_token_id

        if self.pad_to_multiple_of is not None and labels.shape[-1] % self.pad_to_multiple_of != 0:
            pad_len = self.pad_to_multiple_of - labels.shape[-1] % self.pad_to_multiple_of
            labels = F.pad(labels, (0, pad_len), value=pad_token_id)

        labels = labels.masked_fill(labels == pad_token_id, -100)
        batch["labels"] = labels
        return batch


def _load_processed_dataset(path: Path) -> DatasetDict:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run scripts/prepare_dataset.py first."
        )
    dataset = load_from_disk(path.as_posix())
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, received {type(dataset)}")
    return dataset


def _prepare_dataset(
    dataset: DatasetDict,
    processor: AutoProcessor,
    architecture: str,
    num_proc: int = 4,
) -> DatasetDict:
    architecture = architecture.lower()
    remove_columns = next(iter(dataset.values())).column_names

    if architecture == 'whisper':
        feature_extractor = processor.feature_extractor
        tokenizer = processor.tokenizer

        def _prepare_fn(example: Dict) -> Dict:
            audio = example['audio']
            features = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate'])
            labels = tokenizer(example['transcript'])
            example['input_features'] = features['input_features'][0]
            example['labels'] = labels['input_ids']
            return example

    else:
        def _prepare_fn(example: Dict) -> Dict:
            audio = example['audio']
            inputs = processor(audio['array'], sampling_rate=audio['sampling_rate'])
            with processor.as_target_processor():
                labels = processor(example['transcript'])
            example['input_values'] = inputs['input_values'][0]
            example['labels'] = labels['input_ids']
            return example

    vectorized = dataset.map(
        _prepare_fn,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc='Vectorizing dataset',
    )
    return vectorized


def _build_metric_fn(processor: AutoProcessor, architecture: str) -> Callable:
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if architecture == 'whisper':
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        else:
            predicted_ids = np.argmax(predictions, axis=-1)
            decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

        labels = np.where(labels == -100, pad_token_id, labels)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        if not decoded_labels:
            return {'wer': float('nan')}

        score = wer(decoded_labels, decoded_preds)
        return {'wer': score}

    return compute_metrics


def _configure_model(model, processor: AutoProcessor, cfg: ExperimentConfig) -> None:
    architecture = cfg.model.architecture.lower()

    if cfg.model.freeze_encoder:
        if architecture == 'whisper' and hasattr(model, 'model'):
            for param in model.model.encoder.parameters():
                param.requires_grad = False
        elif hasattr(model, 'freeze_feature_encoder'):
            model.freeze_feature_encoder()

    if cfg.model.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False

    if architecture == 'whisper':
        lang = cfg.model.language
        task = cfg.model.task
        if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'set_prefix_tokens'):
            processor.tokenizer.set_prefix_tokens(language=lang, task=task)
        if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'get_decoder_prompt_ids'):
            model.config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(language=lang, task=task)
        if hasattr(model.config, 'suppress_tokens'):
            model.config.suppress_tokens = []

    if architecture != 'whisper' and hasattr(model.config, 'ctc_zero_infinity'):
        model.config.ctc_zero_infinity = True


def _build_training_arguments(cfg: ExperimentConfig, architecture: str, do_eval: bool) -> TrainingArguments:
    dataloader_workers = cfg.dataset.prepare_kwargs.get('num_workers', 2)

    common_kwargs = dict(
        output_dir=cfg.training.output_dir.as_posix(),
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_strategy='steps',
        logging_steps=50,
        fp16=cfg.training.fp16,
        seed=cfg.training.seed,
        dataloader_num_workers=dataloader_workers,
        remove_unused_columns=False,
        report_to=['tensorboard'],
    )

    if do_eval:
        eval_kwargs = dict(
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='wer',
            greater_is_better=False,
        )
    else:
        eval_kwargs = dict(
            evaluation_strategy='no',
            save_strategy='no',
            load_best_model_at_end=False,
        )

    if architecture == 'whisper':
        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            generation_max_length=225,
            **common_kwargs,
            **eval_kwargs,
        )
    else:
        training_args = TrainingArguments(
            **common_kwargs,
            **eval_kwargs,
        )

    return training_args


def run_experiment(cfg: ExperimentConfig) -> None:
    init_logging()
    LOGGER.info(
        "Starting training for architecture=%s, dataset=%s",
        cfg.model.architecture,
        cfg.dataset.name,
    )

    set_seed(cfg.training.seed)

    processed_dataset = _load_processed_dataset(cfg.dataset.processed_dir)

    architecture = cfg.model.architecture.lower()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, processor = load_model_and_processor(
        architecture=cfg.model.architecture,
        pretrained_name=cfg.model.pretrained_name,
        device=device,
    )

    _configure_model(model, processor, cfg)

    num_proc = cfg.dataset.prepare_kwargs.get('num_workers', 2)
    vectorized_dataset = _prepare_dataset(
        processed_dataset,
        processor=processor,
        architecture=architecture,
        num_proc=num_proc,
    )

    train_dataset: Dataset = vectorized_dataset.get('train')
    if train_dataset is None:
        raise ValueError('Training split "train" not found in processed dataset.')

    eval_dataset: Optional[Dataset] = vectorized_dataset.get('validation') or vectorized_dataset.get('test')

    if architecture == 'whisper':
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            pad_to_multiple_of=16,
        )
    else:
        data_collator = DataCollatorCTCWithPadding(processor=processor)

    metric_fn = _build_metric_fn(processor, architecture) if eval_dataset is not None else None
    training_args = _build_training_arguments(cfg, architecture, do_eval=eval_dataset is not None)

    trainer_cls = Seq2SeqTrainer if architecture == 'whisper' else Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if eval_dataset is not None:
        trainer_kwargs['eval_dataset'] = eval_dataset
        trainer_kwargs['compute_metrics'] = metric_fn

    if hasattr(processor, 'tokenizer'):
        trainer_kwargs['tokenizer'] = processor.tokenizer
    if architecture == 'whisper' and hasattr(processor, 'feature_extractor'):
        trainer_kwargs['feature_extractor'] = processor.feature_extractor

    trainer = trainer_cls(**trainer_kwargs)

    trainer.train()
    trainer.save_model()
    if hasattr(trainer, 'save_state'):
        trainer.save_state()

    LOGGER.info('Training complete. Model artifacts saved to %s', cfg.training.output_dir)


__all__ = ['run_experiment']
