# Vietnamese ASR + Deepfake Project

This repository delivers an end-to-end Vietnamese speech-to-text workflow: dataset preparation, fine-tuning state-of-the-art models, and offline or streaming inference. Directories for future deepfake generation/detection work are pre-created so they can be filled in later.

## Highlights
- Unified configuration system for datasets, models, training, and evaluation via Pydantic + YAML.
- Dataset builders for VIVOS, Common Voice (vi), and custom JSON manifests.
- Training pipeline on top of Hugging Face Transformers (Whisper, Wav2Vec2, ...).
- Command line utilities for dataset prep, training, batch transcription, streaming over WebSocket, and synthetic TTS generation.

## Requirements
- Python 3.10+
- NVIDIA GPU with CUDA (strongly recommended for Whisper/Wav2Vec2 fine-tuning)
- fmpeg installed for broad audio format support (recommended)

Install dependencies:

`ash
pip install -r requirements.txt
`

## Repository layout
- sr/config.py � experiment configuration schemas.
- sr/data/ � dataset registry and builders.
- sr/training/trainer.py � fine-tuning entry point using Trainer/Seq2SeqTrainer.
- sr/inference/ � inference helpers and streaming transcriber.
- configs/ � example experiment configs (see whisper_small_vivos.yaml).
- scripts/ � command line utilities for data prep, training, inference, serving, and TTS bootstrapping.
- deepfake/, detect/ � placeholders for deepfake related modules.

## Typical workflow

### 1. Prepare data
`ash
python scripts/prepare_dataset.py run configs/whisper_small_vivos.yaml
`
Options:
- --skip-download to reuse datasets already stored on disk.
- --dataset-name to override the builder specified in the config.

### 2. Fine-tune
`ash
python scripts/train.py run configs/whisper_small_vivos.yaml
`
Use --output-dir to send checkpoints and logs to a custom location.

### 3. Offline transcription
`ash
python scripts/transcribe.py file path/to/audio.wav --config configs/whisper-small-vivos.yaml
`
Useful flags:
- --timestamps to print chunk timestamps (Whisper only).
- --chunk-length-s to control long-form decoding windows.
- Without --config, pass --architecture and --pretrained-name manually.

### 4. Streaming WebSocket service
`ash
python scripts/serve-streaming.py run configs/whisper-small-vivos.yaml --port 9000
`
Endpoints:
- GET /health returns {"status": "ok"}.
- WS /ws/transcribe expects successive PCM32 mono chunks at the model sampling rate (16 kHz by default). Responses contain partial transcripts as JSON; when the client disconnects a final transcript is sent if available.

## Custom datasets
Provide per-split JSON manifests with udio-filepath, 	ext, and optionally speaker-id. Point dataset.name = custom in your config and set prepare_kwargs.manifests to the manifest paths.

### Synthetic TTS bootstrap
When you need quick synthetic data, place sentences in sentences.txt (one line per sentence) and run:
`ash
python scripts/generate_tts-dataset.py run sentences.txt --output-dir data/synthetic_tts --split train
`
The script synthesizes WAV files with pyttsx3, resamples them to 16 kHz, and saves a manifest JSON compatible with the custom dataset builder.

## Next steps
- Flesh out deepfake/ and detect/ for generation/detection research.
- Add denoising, VAD, better chunk policies, and evaluation dashboards.
- Integrate automated testing and CI.

## Data licensing
- VIVOS: released by AIVIVN.
- Common Voice: CC-0.
- User data: ensure you have legal rights before training.

## Support
Logs are written inside 
uns/.... You can increase verbosity with LOGLEVEL=DEBUG before running CLI commands.
