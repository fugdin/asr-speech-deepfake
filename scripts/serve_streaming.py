from pathlib import Path
from typing import Optional

import numpy as np
import typer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from asr.config import ExperimentConfig
from asr.inference import StreamingTranscriber, build_asr_pipeline
from asr.utils import init_logging

app = typer.Typer(help='Serve a streaming ASR WebSocket endpoint.')


def _create_app(transcriber_factory):
    fastapi_app = FastAPI(title='Vietnamese ASR Streaming API')

    @fastapi_app.get('/health')
    async def health() -> JSONResponse:
        return JSONResponse({'status': 'ok'})

    @fastapi_app.websocket('/ws/transcribe')
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        transcriber = transcriber_factory()
        try:
            while True:
                chunk_bytes = await websocket.receive_bytes()
                audio = np.frombuffer(chunk_bytes, dtype=np.float32)
                transcripts = transcriber.append(audio, transcriber.sample_rate)
                if transcripts:
                    await websocket.send_json({'type': 'partial', 'text': ' '.join(transcripts)})
        except WebSocketDisconnect:
            final = transcriber.flush()
            if final:
                try:
                    await websocket.send_json({'type': 'final', 'text': final})
                except RuntimeError:
                    pass
        finally:
            await websocket.close()

    return fastapi_app


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, help='Path to experiment config'),
    host: str = typer.Option('0.0.0.0', help='Bind address for the server'),
    port: int = typer.Option(8000, help='Port for the server'),
    device: Optional[str] = typer.Option(None, help='Device string for the model (cuda|cpu)'),
    chunk_length_s: float = typer.Option(5.0, help='Chunk length for streaming decoding'),
    stride_s: float = typer.Option(1.0, help='Hop length between chunks'),
) -> None:
    init_logging()
    cfg = ExperimentConfig.from_yaml(config)

    asr_pipeline, metadata = build_asr_pipeline(
        architecture=cfg.model.architecture,
        pretrained_name=cfg.model.pretrained_name,
        device=device,
    )

    sample_rate = int(metadata.get('sampling_rate', 16000))
    generate_kwargs = {'language': cfg.model.language, 'task': cfg.model.task} if cfg.model.architecture.lower() == 'whisper' else {}

    def factory():
        return StreamingTranscriber(
            pipeline=asr_pipeline,
            sample_rate=sample_rate,
            chunk_length_s=chunk_length_s,
            stride_s=stride_s,
            generate_kwargs=generate_kwargs,
        )

    fastapi_app = _create_app(factory)
    uvicorn.run(fastapi_app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    app()
