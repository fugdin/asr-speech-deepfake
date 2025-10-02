from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import typer

app = typer.Typer(help="Tao manifest JSON tu du lieu VIVOS da giai nen.")

PROMPTS_FILENAME = "prompts.txt"


def _load_prompts(prompt_file: Path) -> Dict[str, str]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Khong thay file prompts: {prompt_file}")

    mapping: Dict[str, str] = {}
    for line in prompt_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        utt_id, text = parts
        mapping[utt_id] = text.strip()
    return mapping


def _collect_split(split_dir: Path) -> List[Dict[str, str]]:
    audio_root = split_dir / "waves"
    if not audio_root.exists():
        raise FileNotFoundError(f"Thieu thu muc waves trong {split_dir}")

    prompts = _load_prompts(split_dir / PROMPTS_FILENAME)

    records: List[Dict[str, str]] = []
    for wav_path in audio_root.rglob("*.wav"):
        relative = wav_path.relative_to(audio_root)
        utt_id = wav_path.stem
        transcript = prompts.get(utt_id)
        if transcript is None:
            typer.echo(f"Bo qua vi khong tim thay transcript cho {utt_id}")
            continue
        speaker_id = relative.parts[0]
        records.append(
            {
                "audio_filepath": str(wav_path.resolve()),
                "text": transcript,
                "speaker_id": speaker_id,
            }
        )
    return records


@app.command()
def run(
    vivos_root: Path = typer.Argument(..., exists=True, help="Thu muc VIVOS sau khi giai nen"),
    output_dir: Path = typer.Option(Path("data/manifests/vivos"), help="Noi luu manifest"),
) -> None:
    vivos_root = vivos_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split_name in ("train", "test", "validation"):
        split_dir = vivos_root / split_name
        if not split_dir.exists():
            continue

        typer.echo(f"Dang xu ly split '{split_name}'...")
        records = _collect_split(split_dir)
        if not records:
            typer.echo(f"Khong co mau nao o {split_dir}, bo qua.")
            continue

        manifest_path = output_dir / f"{split_name}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        typer.echo(f"Split {split_name}: {len(records)} mau -> {manifest_path}")
        total += len(records)

    if total == 0:
        raise typer.Exit(code=1)

    typer.echo("Hoan tat. Co the dung manifest nay voi dataset 'custom'.")


if __name__ == "__main__":
    app()
