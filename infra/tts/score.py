#!/usr/bin/env python3
"""Owned speaker-embedding + WER scorer.

Fail-closes when models are not loaded. Similarity is never RMS or a file hash.
"""

from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from pydantic import BaseModel

SCORE_RUNTIME: dict[str, Any] = {
    "ready": False,
    "error": "score models not loaded",
    "embedder": None,
    "asr": None,
    "embedding_model": None,
    "asr_model": None,
}
REFERENCE_EMBED_CACHE: dict[str, np.ndarray] = {}


class ScoreUnavailable(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ScoreRequest(BaseModel):
    engine: str
    text: str
    reference_audio: str
    hypothesis_audio: str


class EmbedRequest(BaseModel):
    audio: str


class WavLMEmbedder:
    def __init__(self, model: Any, sample_rate: int, device: str):
        self.model = model
        self.sample_rate = sample_rate
        self.device = device

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch
        import torchaudio

        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        if int(sample_rate) != int(self.sample_rate):
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), int(self.sample_rate))
        waveform = waveform.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveform)
            hidden = features[-1]
            vector = hidden.mean(dim=1).squeeze(0).detach().float().cpu().numpy()
        norm = float(np.linalg.norm(vector)) + 1e-12
        return (vector / norm).astype(np.float64)


class Wav2Vec2Asr:
    def __init__(self, model: Any, labels: tuple[str, ...] | list[str], sample_rate: int, device: str):
        self.model = model
        self.labels = list(labels)
        self.sample_rate = sample_rate
        self.device = device
        self.blank = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        import torch
        import torchaudio

        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        if int(sample_rate) != int(self.sample_rate):
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), int(self.sample_rate))
        waveform = waveform.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            emissions, _ = self.model(waveform)
            tokens = torch.argmax(emissions[0], dim=-1).detach().cpu().tolist()
        return _ctc_decode(tokens, self.labels, self.blank)


def reset_score_runtime() -> None:
    REFERENCE_EMBED_CACHE.clear()
    SCORE_RUNTIME.update(
        {
            "ready": False,
            "error": "score models not loaded",
            "embedder": None,
            "asr": None,
            "embedding_model": None,
            "asr_model": None,
        }
    )


def install_score_runtime(
    *,
    embedder: Any,
    asr: Any,
    embedding_model: str = "injected",
    asr_model: str = "injected",
) -> None:
    REFERENCE_EMBED_CACHE.clear()
    SCORE_RUNTIME.update(
        {
            "ready": True,
            "error": None,
            "embedder": embedder,
            "asr": asr,
            "embedding_model": embedding_model,
            "asr_model": asr_model,
        }
    )


def score_runtime_ready() -> bool:
    return bool(SCORE_RUNTIME["ready"] and SCORE_RUNTIME["embedder"] is not None and SCORE_RUNTIME["asr"] is not None)


def score_model_home() -> Path:
    raw = os.environ.get("OWNED_TTS_SCORE_MODEL_DIR", "").strip()
    if raw:
        return Path(raw)
    return Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch")))


def load_score_models() -> None:
    home = score_model_home()
    home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(home)
    try:
        import torch
        import torchaudio

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embed_bundle = torchaudio.pipelines.WAVLM_BASE
        asr_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        embed_model = embed_bundle.get_model().to(device).eval()
        asr_model = asr_bundle.get_model().to(device).eval()
        install_score_runtime(
            embedder=WavLMEmbedder(embed_model, embed_bundle.sample_rate, device),
            asr=Wav2Vec2Asr(asr_model, asr_bundle.get_labels(), asr_bundle.sample_rate, device),
            embedding_model="wavlm-base",
            asr_model="wav2vec2-asr-base-960h",
        )
    except Exception as exc:
        reset_score_runtime()
        SCORE_RUNTIME["error"] = str(exc)
        raise ScoreUnavailable("SCORE_MODEL_MISSING", f"failed to load speaker-embedding/ASR models: {exc}") from exc


def _require_ready() -> None:
    if not score_runtime_ready():
        raise ScoreUnavailable(
            "SCORE_MODEL_MISSING",
            SCORE_RUNTIME.get("error") or "speaker-embedding/ASR models are not loaded",
        )


def decode_audio_b64(raw: str) -> tuple[np.ndarray, int]:
    payload = raw.strip()
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        wav = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ScoreUnavailable("SCORE_AUDIO_INVALID", f"audio was not valid base64: {exc}") from exc
    try:
        audio, sample_rate = sf.read(io.BytesIO(wav))
    except Exception as exc:
        raise ScoreUnavailable("SCORE_AUDIO_INVALID", f"audio was not a readable WAV: {exc}") from exc
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0 or int(sample_rate) <= 0:
        raise ScoreUnavailable("SCORE_AUDIO_INVALID", "audio is empty")
    return samples, int(sample_rate)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        raise ScoreUnavailable("SCORE_MODEL_MISSING", "speaker embeddings were empty or mismatched")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def normalize_words(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()
    return [word for word in cleaned.split() if word]


def word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    ref = normalize_words(reference_text)
    hyp = normalize_words(hypothesis_text)
    if not ref:
        return 0.0 if not hyp else 1.0
    rows = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    rows[:, 0] = np.arange(len(ref) + 1)
    rows[0, :] = np.arange(len(hyp) + 1)
    for i, ref_word in enumerate(ref, start=1):
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            rows[i, j] = min(rows[i - 1, j] + 1, rows[i, j - 1] + 1, rows[i - 1, j - 1] + cost)
    return float(rows[len(ref), len(hyp)] / len(ref))


def _ctc_decode(tokens: list[int], labels: list[str], blank: int) -> str:
    chars: list[str] = []
    prev = None
    for token in tokens:
        if token == blank or token == prev:
            prev = token
            continue
        if 0 <= token < len(labels):
            chars.append(labels[token])
        prev = token
    transcript = "".join(chars).replace("|", " ")
    return re.sub(r"\s+", " ", transcript).strip()


def pitch_variation(audio: np.ndarray, sample_rate: int) -> float:
    samples = np.asarray(audio, dtype=np.float64).ravel()
    if samples.size < sample_rate // 4:
        return 0.0
    frame = max(int(sample_rate * 0.04), 64)
    hop = max(frame // 2, 32)
    min_lag = max(int(sample_rate / 400), 2)
    max_lag = min(int(sample_rate / 60), frame - 2)
    if max_lag <= min_lag:
        return 0.0
    f0s: list[float] = []
    window = np.hanning(frame)
    for start in range(0, samples.size - frame, hop):
        chunk = samples[start : start + frame] * window
        if float(np.sqrt(np.mean(chunk * chunk))) < 0.01:
            continue
        corr = np.correlate(chunk, chunk, mode="full")[frame - 1 :]
        lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
        if corr[lag] <= 0:
            continue
        f0s.append(float(sample_rate) / float(lag))
    if len(f0s) < 3:
        return 0.0
    mean = float(np.mean(f0s))
    if mean <= 0:
        return 0.0
    return float(np.clip(np.std(f0s) / mean, 0.0, 1.0))


def embed_audio(audio: np.ndarray, sample_rate: int, cache_key: str | None = None) -> np.ndarray:
    _require_ready()
    if cache_key:
        cached = REFERENCE_EMBED_CACHE.get(cache_key)
        if cached is not None:
            return cached
    vector = np.asarray(SCORE_RUNTIME["embedder"].embed(audio, sample_rate), dtype=np.float64)
    if cache_key:
        if len(REFERENCE_EMBED_CACHE) >= 8:
            REFERENCE_EMBED_CACHE.pop(next(iter(REFERENCE_EMBED_CACHE)))
        REFERENCE_EMBED_CACHE[cache_key] = vector
    return vector


def transcribe_audio(audio: np.ndarray, sample_rate: int) -> str:
    _require_ready()
    return str(SCORE_RUNTIME["asr"].transcribe(audio, sample_rate))


def score_request(payload: ScoreRequest | dict[str, Any]) -> dict[str, Any]:
    _require_ready()
    request = payload if isinstance(payload, ScoreRequest) else ScoreRequest.model_validate(payload)
    reference, ref_rate = decode_audio_b64(request.reference_audio)
    hypothesis, hyp_rate = decode_audio_b64(request.hypothesis_audio)
    similarity = cosine_similarity(
        embed_audio(reference, ref_rate, cache_key=request.reference_audio),
        embed_audio(hypothesis, hyp_rate),
    )
    transcript = transcribe_audio(hypothesis, hyp_rate)
    return {
        "engine": request.engine,
        "similarity": similarity,
        "wer": word_error_rate(request.text, transcript),
        "expressiveness": pitch_variation(hypothesis, hyp_rate),
        "similarity_method": "speaker-embedding",
        "asr_transcript": transcript,
        "embedding_model": SCORE_RUNTIME["embedding_model"],
        "asr_model": SCORE_RUNTIME["asr_model"],
    }


def embed_request(audio_b64: str) -> dict[str, Any]:
    _require_ready()
    audio, sample_rate = decode_audio_b64(audio_b64)
    vector = embed_audio(audio, sample_rate)
    return {
        "embedding": vector.astype(float).tolist(),
        "model": SCORE_RUNTIME["embedding_model"],
        "method": "speaker-embedding",
    }


def maybe_load_score_models() -> None:
    if os.environ.get("OWNED_TTS_SKIP_SCORE_LOAD") == "1":
        return
    model_dir = os.environ.get("OWNED_TTS_SCORE_MODEL_DIR", "").strip()
    if not model_dir and os.environ.get("OWNED_TTS_LOAD_SCORE") != "1":
        return
    try:
        load_score_models()
    except ScoreUnavailable:
        # Leave SCORE_RUNTIME unready. /v1/score stays 503; synthesize is unaffected.
        return
