from pathlib import Path
from typing import Optional

import typer

from asr.inference import transcribe_file

app = typer.Typer(help='Transcribe Vietnamese speech recordings.')


@app.command()
def file(
    audio: Path = typer.Argument(..., exists=True, help='Input audio file path'),
    config: Optional[Path] = typer.Option(None, exists=True, help='Experiment config to load model info'),
    architecture: Optional[str] = typer.Option(None, help='Model architecture (whisper|wav2vec2|...)'),
    pretrained_name: Optional[str] = typer.Option(None, help='Pretrained checkpoint name'),
    device: Optional[str] = typer.Option(None, help='Device string, e.g. cuda or cpu'),
    chunk_length_s: float = typer.Option(30.0, help='Chunk length in seconds for long-form decoding'),
    timestamps: bool = typer.Option(False, help='Return word-level timestamps when available'),
    output: Optional[Path] = typer.Option(None, help='Save transcription to this file'),
) -> None:
    result = transcribe_file(
        audio_path=audio,
        config_path=config,
        architecture=architecture,
        pretrained_name=pretrained_name,
        device=device,
        chunk_length_s=chunk_length_s,
        return_timestamps=timestamps,
    )

    transcript = result['text']
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(transcript, encoding='utf-8')
    else:
        typer.echo(transcript)

    if timestamps and 'chunks' in result and not output:
        typer.echo('\n--- segments ---')
        for chunk in result['chunks']:
            start, end = chunk.get('timestamp', (None, None))
            start_val = float(start) if start is not None else 0.0
            end_val = float(end) if end is not None else 0.0
            typer.echo(f"[{start_val:.2f}-{end_val:.2f}] {chunk.get('text', '').strip()}")


if __name__ == '__main__':
    app()
