#!/usr/bin/env python3
"""Owned dots.tts worker used by Beam and Hugging Face Endpoints."""

from __future__ import annotations

import base64
import binascii
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from score import EmbedRequest, ScoreRequest, ScoreUnavailable, embed_request, maybe_load_score_models, score_request

SAMPLE_RATE = 48000
MODEL_NAME_BY_VARIANT = {
    "soar": "dots-studio/dots.tts-soar",
    "mf": "dots-studio/dots.tts-mf",
}
CHECKPOINT_MARKERS = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")

app = FastAPI(title="owned-dots-tts", version="1.1.0")
RUNTIME: dict[str, Any] = {
    "engine": None,
    "loaded_variant": None,
    "ready": False,
    "error": "runtime not loaded",
    "cold": True,
    "model_name": None,
}


class SynthesizeRequest(BaseModel):
    text: str
    persona: str = "mistress_mandy"
    style: str = "neutral"
    model: str = "dots.tts-soar"
    variant: str = "soar"
    prompt_audio: str | None = None
    prompt_text: str | None = None
    sample_rate: int = SAMPLE_RATE
    model_name_or_path: str = Field(default="dots-studio/dots.tts-soar")
    num_steps: int = 10
    guidance_scale: float = 1.2
    gpu: str | None = None


class DotsTtsUnavailable(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def measure_audio(samples: Any, sample_rate: int) -> dict[str, float]:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float64).ravel()
    if audio.size == 0 or sample_rate <= 0:
        return {"duration": 0.0, "rms": 0.0, "peak_hz": 0.0, "speech_band": 0.0, "low_band": 0.0}
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / float(sample_rate))
    total = float(spectrum[1:].sum()) + 1e-12
    peak_hz = float(freqs[int(np.argmax(spectrum[1:])) + 1]) if spectrum.size > 1 else 0.0
    return {
        "duration": float(audio.size / float(sample_rate)),
        "rms": float((audio**2).mean() ** 0.5),
        "peak_hz": peak_hz,
        "speech_band": float(spectrum[(freqs >= 300.0) & (freqs < 4000.0)].sum() / total),
        "low_band": float(spectrum[(freqs >= 80.0) & (freqs < 250.0)].sum() / total),
    }


def amplify_quiet_speechlike(samples: Any, sample_rate: int) -> Any:
    import numpy as np

    metrics = measure_audio(samples, sample_rate)
    if metrics["speech_band"] < 0.25 or metrics["rms"] >= 0.01 or metrics["rms"] <= 0:
        return samples
    audio = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    gain = min(0.89 / peak, 0.02 / metrics["rms"])
    return audio * gain


def is_speechlike(
    samples: Any,
    sample_rate: int,
    *,
    min_duration: float = 0.5,
    min_speech_band: float = 0.25,
    min_rms: float = 0.01,
) -> bool:
    metrics = measure_audio(samples, sample_rate)
    if metrics["duration"] < min_duration or metrics["rms"] < min_rms:
        return False
    if metrics["speech_band"] < min_speech_band:
        return False
    # 185 Hz sine rumble has almost no 300-4000 Hz energy. A voiced F0 in the
    # same bin still has formants, so do not reject on peak_hz alone.
    if metrics["low_band"] >= 0.70 and metrics["speech_band"] < 0.40:
        return False
    return True


def _model_name(variant: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return os.environ.get("OWNED_TTS_MODEL_NAME") or MODEL_NAME_BY_VARIANT.get(variant, "dots-studio/dots.tts-soar")


def _configured_model_dir() -> Path | None:
    raw = os.environ.get("OWNED_TTS_MODEL_DIR", "").strip()
    return Path(raw) if raw else None


def _dir_has_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {child.name for child in path.iterdir()}
    return any(marker in names for marker in CHECKPOINT_MARKERS)


def resolve_checkpoint(variant: str, explicit: str | None = None) -> str:
    model_dir = _configured_model_dir()
    if model_dir is not None and _dir_has_checkpoint(model_dir):
        return str(model_dir)
    model_name = _model_name(variant, explicit)
    local = Path(model_name)
    if local.exists() and _dir_has_checkpoint(local):
        return str(local)
    if os.environ.get("OWNED_TTS_ALLOW_DOWNLOAD") == "1":
        return model_name
    if model_dir is not None:
        raise DotsTtsUnavailable(
            "CHECKPOINT_MISSING",
            f"OWNED_TTS_MODEL_DIR has no dots.tts weights: {model_dir}",
        )
    raise DotsTtsUnavailable(
        "CHECKPOINT_MISSING",
        "dots.tts checkpoint is not present; set OWNED_TTS_MODEL_DIR or OWNED_TTS_ALLOW_DOWNLOAD=1",
    )


def import_dots_runtime():
    try:
        from dots_tts.runtime import DotsTtsRuntime  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional GPU image
        raise DotsTtsUnavailable("ENGINE_UNAVAILABLE", f"dots_tts import failed: {exc}") from exc
    return DotsTtsRuntime


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def reported_device_fields(
    *,
    requested_gpu: str | None,
    device: str | None = None,
    cuda_available: bool | None = None,
) -> dict[str, str | None]:
    if cuda_available is None:
        cuda_available = _cuda_available()
    runtime_device = device or ("cuda" if cuda_available else "cpu")
    if runtime_device != "cuda":
        gpu = "cpu"
    else:
        gpu = requested_gpu or "cuda"
    return {
        "device": runtime_device,
        "gpu": gpu,
        "requested_gpu": requested_gpu,
    }


def select_runtime_precision(*, cuda_available: bool | None = None) -> str:
    explicit = os.environ.get("OWNED_TTS_PRECISION", "").strip()
    if explicit:
        return explicit
    if cuda_available is None:
        cuda_available = _cuda_available()
    return "bfloat16" if cuda_available else "float32"


def select_runtime_optimize(*, cuda_available: bool | None = None) -> bool:
    explicit = os.environ.get("OWNED_TTS_OPTIMIZE", "").strip()
    if explicit:
        return explicit == "1"
    if cuda_available is None:
        cuda_available = _cuda_available()
    return bool(cuda_available)


def runtime_status() -> dict[str, Any]:
    model_dir = _configured_model_dir()
    variant = str(RUNTIME.get("loaded_variant") or os.environ.get("OWNED_TTS_VARIANT", "soar"))
    requested_gpu = os.environ.get("OWNED_TTS_GPU", "").strip() or (
        "A10G" if variant == "mf" else "A100-40GB"
    )
    devices = reported_device_fields(requested_gpu=requested_gpu)
    payload = {
        "engine": "dots.tts",
        "ready": bool(RUNTIME.get("ready") and RUNTIME.get("engine") is not None),
        "model_name": RUNTIME.get("model_name") or _model_name("soar"),
        "model_dir": str(model_dir) if model_dir else None,
        "error": RUNTIME.get("error"),
        "gpu": devices["gpu"],
        "device": devices["device"],
        "requested_gpu": devices["requested_gpu"],
        "keep_warm_seconds": int(os.environ.get("OWNED_TTS_KEEP_WARM_SECONDS", "300")),
        "queue": {
            "max": int(os.environ.get("OWNED_TTS_MAX_QUEUE", "32")),
            "in_flight": 0,
        },
    }
    if payload["ready"]:
        payload["status"] = "ok"
        payload["error"] = None
        return payload
    try:
        import_dots_runtime()
        resolve_checkpoint(os.environ.get("OWNED_TTS_VARIANT", "soar"))
        payload["status"] = "uninitialized"
        payload["error"] = payload["error"] or "dots.tts weights are present but the engine is not loaded"
    except DotsTtsUnavailable as exc:
        payload["status"] = "unavailable"
        payload["error"] = exc.message
        payload["code"] = exc.code
    return payload


def _load_runtime(variant: str, explicit_model: str | None = None) -> Any:
    if RUNTIME["engine"] is not None and RUNTIME["loaded_variant"] == variant and RUNTIME["ready"]:
        return RUNTIME["engine"]
    runtime_cls = import_dots_runtime()
    model_name = resolve_checkpoint(variant, explicit_model)
    cuda_available = _cuda_available()
    precision = select_runtime_precision(cuda_available=cuda_available)
    optimize = select_runtime_optimize(cuda_available=cuda_available)
    max_generate_length = int(os.environ.get("OWNED_TTS_MAX_AUDIO_PATCHES", "48"))
    engine = runtime_cls.from_pretrained(
        model_name,
        precision=precision,
        optimize=optimize,
        max_generate_length=max_generate_length,
    )
    if not cuda_available:
        try:
            import torch

            torch.set_num_threads(max(1, int(os.environ.get("OWNED_TTS_CPU_THREADS", os.cpu_count() or 1))))
        except Exception:
            pass
    RUNTIME.update(
        {
            "engine": engine,
            "loaded_variant": variant,
            "ready": True,
            "error": None,
            "model_name": model_name,
            "precision": precision,
            "device": "cuda" if cuda_available else "cpu",
        }
    )
    return engine


def warmup_runtime(variant: str | None = None) -> dict[str, Any]:
    chosen = variant or os.environ.get("OWNED_TTS_VARIANT", "soar")
    try:
        _load_runtime(chosen)
    except DotsTtsUnavailable as exc:
        RUNTIME["ready"] = False
        RUNTIME["error"] = exc.message
        RUNTIME["code"] = exc.code
    return runtime_status()


def _write_prompt_audio(prompt_audio: str | None) -> str | None:
    if not prompt_audio:
        return None
    if not prompt_audio.startswith("data:") and len(prompt_audio) < 4096 and Path(prompt_audio).is_file():
        return prompt_audio
    raw = prompt_audio
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        blob = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        raise DotsTtsUnavailable("REFERENCE_INVALID", "prompt_audio is not a readable path or WAV payload")
    if len(blob) < 44 or blob[:4] != b"RIFF":
        raise DotsTtsUnavailable("REFERENCE_INVALID", "prompt_audio must be a RIFF/WAVE payload")
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.write(blob)
    handle.close()
    return handle.name


def _require_speechlike_wav(path: str, *, role: str) -> dict[str, float]:
    import soundfile as sf

    samples, rate = sf.read(path)
    metrics = measure_audio(samples, int(rate))
    min_duration = 0.5 if role == "prompt" else 1.0
    if not is_speechlike(samples, int(rate), min_duration=min_duration):
        code = "REFERENCE_INVALID" if role == "prompt" else "ENGINE_UNAVAILABLE"
        raise DotsTtsUnavailable(
            code,
            f"{role} audio is not speechlike: duration={metrics['duration']:.2f}s "
            f"speech_band={metrics['speech_band']:.3f} peak_hz={metrics['peak_hz']:.1f}",
        )
    return metrics



def _ensure_min_audio_seconds(samples: Any, sample_rate: int, *, min_seconds: float = 1.05) -> Any:
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32).ravel()
    need = max(1, int(round(float(min_seconds) * int(sample_rate))))
    if audio.size >= need:
        return samples
    padded = np.zeros(need, dtype=np.float32)
    padded[: audio.size] = audio
    return padded


def apply_max_generate_length(engine: Any, length: int) -> None:
    if engine is None or not hasattr(engine, "max_generate_length"):
        return
    try:
        engine.max_generate_length = max(1, int(length))
    except Exception:
        return


def requested_max_generate_length(payload: SynthesizeRequest | None = None) -> int | None:
    raw = os.environ.get("OWNED_TTS_MAX_AUDIO_PATCHES", "").strip()
    if raw:
        return max(1, int(raw))
    extra = getattr(payload, "max_generate_length", None) if payload is not None else None
    if extra:
        return max(1, int(extra))
    return None


def synthesize(payload: SynthesizeRequest) -> dict[str, Any]:
    started = time.perf_counter()
    prompt_path = _write_prompt_audio(payload.prompt_audio)
    if prompt_path:
        _require_speechlike_wav(prompt_path, role="prompt")
    engine = _load_runtime(payload.variant, payload.model_name_or_path)
    cap = requested_max_generate_length(payload)
    if cap is not None:
        apply_max_generate_length(engine, cap)
    cold_start = RUNTIME["cold"]
    RUNTIME["cold"] = False
    generate_kwargs: dict[str, Any] = {
        "text": payload.text,
        "template_name": "tts",
        "language": os.environ.get("OWNED_TTS_LANGUAGE", "en"),
        "num_steps": payload.num_steps,
        "guidance_scale": payload.guidance_scale,
        "normalize_text": True,
    }
    if prompt_path:
        generate_kwargs["prompt_audio_path"] = prompt_path
        generate_kwargs["prompt_text"] = payload.prompt_text
    result = engine.generate(**generate_kwargs)
    audio = result["audio"].float().cpu().squeeze().numpy()
    sample_rate = int(result.get("sample_rate") or payload.sample_rate or SAMPLE_RATE)
    audio = amplify_quiet_speechlike(audio, sample_rate)
    audio = _ensure_min_audio_seconds(audio, sample_rate, min_seconds=1.05)
    generated = measure_audio(audio, sample_rate)
    if generated["duration"] < 1.0:
        _require_speechlike_wav_from_samples(audio, sample_rate)
    buffer = io.BytesIO()
    import soundfile as sf

    sf.write(buffer, audio, sample_rate, format="WAV")
    wav = buffer.getvalue()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "audio_base64": base64.b64encode(wav).decode("ascii"),
        "model": payload.model,
        "gpu_time_ms": elapsed_ms,
        "audio_seconds": max(0.05, len(audio) / float(sample_rate)),
        "cold_start": cold_start,
        "sample_rate": sample_rate,
        **reported_device_fields(
            requested_gpu=payload.gpu,
            device=str(RUNTIME.get("device") or ""),
        ),
        "provider_container": os.environ.get("OWNED_TTS_CONTAINER_IMAGE", "owned-dots-tts"),
    }


def wants_wav_stream(accept: str) -> bool:
    return "audio/wav" in (accept or "").lower()


def wav_stream_response(body: dict[str, Any]) -> StreamingResponse:
    wav = base64.b64decode(str(body.get("audio_base64") or ""))

    def chunks(step: int = 65536):
        for index in range(0, len(wav), step):
            yield wav[index : index + step]

    return StreamingResponse(
        chunks(),
        media_type="audio/wav",
        headers={
            "x-gpu-time-ms": str(int(body.get("gpu_time_ms") or 0)),
            "x-audio-seconds": str(body.get("audio_seconds") or 0),
            "x-cold-start": "1" if body.get("cold_start") else "0",
            "x-model": str(body.get("model") or ""),
            "x-device": str(body.get("device") or ""),
        },
    )


def _require_speechlike_wav_from_samples(samples: Any, sample_rate: int) -> dict[str, float]:
    samples = amplify_quiet_speechlike(samples, sample_rate)
    metrics = measure_audio(samples, sample_rate)
    if not is_speechlike(samples, sample_rate, min_duration=1.0):
        raise DotsTtsUnavailable(
            "ENGINE_UNAVAILABLE",
            "generated audio is not speechlike: duration="
            f"{metrics['duration']:.2f}s speech_band={metrics['speech_band']:.3f} "
            f"peak_hz={metrics['peak_hz']:.1f} rms={metrics['rms']:.4f}",
        )
    return metrics


@app.on_event("startup")
def _startup_load() -> None:
    maybe_load_score_models()
    if os.environ.get("OWNED_TTS_SKIP_STARTUP_LOAD") == "1":
        return
    warmup_runtime()


@app.get("/healthz")
def healthz() -> JSONResponse:
    payload = runtime_status()
    status_code = 200 if payload.get("status") == "ok" else 503
    return JSONResponse(payload, status_code=status_code)


@app.post("/v1/score")
def score_route(payload: ScoreRequest) -> JSONResponse:
    try:
        return JSONResponse(score_request(payload))
    except ScoreUnavailable as exc:
        return JSONResponse({"error": exc.code, "message": exc.message}, status_code=503)


@app.post("/v1/embed")
def embed_route(payload: EmbedRequest) -> JSONResponse:
    try:
        return JSONResponse(embed_request(payload.audio))
    except ScoreUnavailable as exc:
        return JSONResponse({"error": exc.code, "message": exc.message}, status_code=503)


@app.post("/v1/synthesize")
def synthesize_route(payload: SynthesizeRequest, request: Request):
    try:
        body = synthesize(payload)
        if wants_wav_stream(request.headers.get("accept", "")):
            return wav_stream_response(body)
        return JSONResponse(body)
    except DotsTtsUnavailable as exc:
        if RUNTIME.get("engine") is None:
            RUNTIME["ready"] = False
            RUNTIME["error"] = exc.message
        return JSONResponse({"error": exc.code, "message": exc.message, "engine": "dots.tts"}, status_code=503)
    except Exception as exc:  # pragma: no cover - unexpected GPU/runtime failure
        if RUNTIME.get("engine") is None:
            RUNTIME["ready"] = False
            RUNTIME["error"] = str(exc)
        return JSONResponse(
            {"error": "ENGINE_UNAVAILABLE", "message": str(exc), "engine": "dots.tts"},
            status_code=503,
        )


_startup_load()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
