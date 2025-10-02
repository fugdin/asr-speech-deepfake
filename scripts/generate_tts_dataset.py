from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import pyttsx3
import soundfile as sf
import typer


app = typer.Typer(help="Sinh du lieu TTS tieng Viet va manifest JSON cho pipeline ASR.")


def _read_sentences(source: Path) -> List[str]:
    text = source.read_text(encoding='utf-8')
    sentences = [line.strip() for line in text.splitlines() if line.strip()]
    if not sentences:
        raise typer.BadParameter("File cau dau vao rong.")
    return sentences


def _select_voice(engine: pyttsx3.Engine, keyword: Optional[str]) -> None:
    if not keyword:
        return
    keyword = keyword.lower()
    for voice in engine.getProperty('voices'):
        descriptor = f"{voice.id} {voice.name}".lower()
        if keyword in descriptor:
            engine.setProperty('voice', voice.id)
            typer.echo(f"Chon voice: {voice.name} ({voice.id})")
            return
    typer.echo("Khong tim thay voice khop, dung voice mac dinh.")


def _synthesize(engine: pyttsx3.Engine, text: str, target_path: Path, sample_rate: int) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    engine.save_to_file(text, tmp_path.as_posix())
    engine.runAndWait()

    waveform, sr = librosa.load(tmp_path.as_posix(), sr=None, mono=True)
    tmp_path.unlink(missing_ok=True)

    if sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    sf.write(target_path.as_posix(), waveform.astype(np.float32), sr)


@app.command()
def run(
    sentences: Path = typer.Argument(..., exists=True, help='File .txt moi dong mot cau'),
    output_dir: Path = typer.Option(Path('data/synthetic_tts'), help='Thu muc luu du lieu TTS'),
    split: str = typer.Option('train', help='Ten split (train/dev/test...)'),
    sample_rate: int = typer.Option(16000, help='Tan so mau muc tieu cho file WAV'),
    voice_keyword: Optional[str] = typer.Option(None, help='Tu khoa tim kiem voice (vi du: "female", "male", "Vietnam"...)'),
    rate: Optional[int] = typer.Option(None, help='Toc do noi (words per minute) pyttsx3'),
    volume: Optional[float] = typer.Option(None, help='Am luong 0.0-1.0'),
    manifest_name: str = typer.Option('manifest.json', help='Ten file manifest duoc xuat trong thu muc split'),
) -> None:
    sentences_list = _read_sentences(sentences)

    engine = pyttsx3.init()
    if rate is not None:
        engine.setProperty('rate', rate)
    if volume is not None:
        engine.setProperty('volume', float(volume))
    _select_voice(engine, voice_keyword)

    split_dir = output_dir / split
    audio_dir = split_dir / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for idx, text in enumerate(sentences_list, start=1):
        filename = f"utt_{idx:05d}.wav"
        target_path = audio_dir / filename
        typer.echo(f"[{idx}/{len(sentences_list)}] -> {target_path}")
        _synthesize(engine, text, target_path, sample_rate)
        entries.append({
            'audio_filepath': str(target_path.resolve()),
            'text': text,
        })

    manifest_path = split_dir / manifest_name
    manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf-8')
    typer.echo(f"Da sinh {len(entries)} mau, manifest: {manifest_path}")


if __name__ == '__main__':
    app()
