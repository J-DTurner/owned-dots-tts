"""Beam ASGI entrypoints for the owned FastAPI worker.

Production is `CMD python /app/worker.py` on this image. Do not deploy a
Beam `@endpoint` function wrapper named `generate`; the API posts to
`/v1/synthesize`, `/v1/score`, `/v1/embed`, and `GET /healthz`.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

try:
    from beam import Image, Volume, asgi
except Exception:  # pragma: no cover - local tests without the Beam SDK
    def asgi(**_kwargs):
        def decorator(fn):
            fn._beam_asgi = _kwargs
            return fn

        return decorator

    class Image:  # type: ignore[no-redef]
        def __init__(self, value: str | None = None):
            self.value = value
            self.python_packages: list[str] = []
            self.build_steps: list[dict[str, str]] = []

        @staticmethod
        def from_dockerfile(path: str):
            return Image(value=path)

        @staticmethod
        def from_id(image_id: str):
            return Image(value=image_id)

        @staticmethod
        def from_registry(image_uri: str, credentials=None):
            return Image(value=image_uri)

        def add_python_packages(self, packages):
            self.python_packages.extend(packages)
            for package in packages:
                self.build_steps.append({"command": package, "type": "pip"})
            return self

    class Volume:  # type: ignore[no-redef]
        def __init__(self, name: str, mount_path: str):
            self.name = name
            self.mount_path = mount_path


KEEP_WARM_SECONDS = int(os.environ.get("OWNED_TTS_KEEP_WARM_SECONDS", "300"))
PREMIUM_GPU = os.environ.get("OWNED_TTS_PREMIUM_GPU", "A100-40GB")
INTERACTIVE_GPU = os.environ.get("OWNED_TTS_INTERACTIVE_GPU", "A10G")
ASGI_MEMORY = os.environ.get("OWNED_TTS_ASGI_MEMORY", "16Gi")
DOCKERFILE_PATH = "infra/tts/Dockerfile"
MODEL_VOLUME = Volume(name="owned-dots-models", mount_path="/models/dots")


def upstream_url() -> str:
    return os.environ.get("OWNED_TTS_UPSTREAM_URL", "").strip().rstrip("/")


def select_worker_image_spec() -> dict[str, str]:
    image_id = os.environ.get("OWNED_TTS_IMAGE_ID", "").strip()
    if image_id:
        return {"kind": "id", "value": image_id}
    image_uri = os.environ.get("OWNED_TTS_IMAGE_URI", "").strip()
    if image_uri:
        return {"kind": "registry", "value": image_uri}
    if upstream_url():
        return {"kind": "registry", "value": "python:3.12-slim"}
    return {"kind": "dockerfile", "value": DOCKERFILE_PATH}


WORKER_PACKAGES = (
    "fastapi>=0.140.0",
    "uvicorn[standard]==0.35.0",
    "gunicorn",
    "soundfile==0.13.1",
    "dots.tts>=0.3.1",
)
PROXY_PACKAGES = (
    "fastapi>=0.140.0",
    "uvicorn[standard]==0.35.0",
    "gunicorn",
    "httpx==0.28.1",
)


def worker_image():
    spec = select_worker_image_spec()
    if spec["kind"] == "id":
        image = Image.from_id(spec["value"])
    elif spec["kind"] == "registry":
        image = Image.from_registry(spec["value"])
    else:
        return Image.from_dockerfile(spec["value"])
    if os.environ.get("OWNED_TTS_SKIP_IMAGE_PACKAGES") == "1":
        return image
    adder = getattr(image, "add_python_packages", None)
    if callable(adder):
        packages = PROXY_PACKAGES if upstream_url() else WORKER_PACKAGES
        return adder(packages)
    return image


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ensure_runtime_packages() -> None:
    needed = ("fastapi", "dots_tts", "soundfile")
    if all(_module_available(name) for name in needed):
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", *WORKER_PACKAGES]
    )


WORKER_IMAGE = worker_image()


def _load_worker():
    ensure_runtime_packages()
    from worker import app as fastapi_app, warmup_runtime as worker_warmup

    return fastapi_app, worker_warmup


def warmup_runtime(*args, **kwargs):
    if upstream_url():
        return None
    _, hook = _load_worker()
    return hook(*args, **kwargs)


def _proxy_app():
    import json
    from fastapi import FastAPI
    from fastapi.responses import Response
    import urllib.error
    import urllib.request

    app = FastAPI()
    base = upstream_url()

    def _forward(path: str, *, method: str = "GET", body: bytes = b"", headers: dict[str, str] | None = None):
        target = f"{base}{path}"
        req_headers = {
            key: value
            for key, value in (headers or {}).items()
            if key.lower() not in {"host", "content-length"}
        }
        req = urllib.request.Request(target, data=body or None, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = resp.read()
                return Response(content=payload, status_code=resp.status, media_type=resp.headers.get("Content-Type"))
        except urllib.error.HTTPError as exc:
            return Response(content=exc.read(), status_code=int(exc.code), media_type=exc.headers.get("Content-Type"))

    @app.get("/healthz")
    def healthz():
        return _forward("/healthz")

    @app.post("/v1/synthesize")
    async def synthesize(payload: dict):
        return _forward(
            "/v1/synthesize",
            method="POST",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    @app.post("/v1/score")
    async def score(payload: dict):
        return _forward(
            "/v1/score",
            method="POST",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    @app.post("/v1/embed")
    async def embed(payload: dict):
        return _forward(
            "/v1/embed",
            method="POST",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    return app


def _asgi_app(_context=None):
    if upstream_url():
        return _proxy_app()
    fastapi_app, _ = _load_worker()
    return fastapi_app


def _gpu_or_none(name: str) -> str | None:
    value = (name or "").strip()
    if not value or value.lower() in {"cpu", "none", "off"}:
        return None
    return value


def _asgi_spec(name: str, *, variant: str, gpu_name: str) -> dict:
    gpu = _gpu_or_none(gpu_name)
    spec = {
        "name": name,
        "cpu": 1,
        "memory": ASGI_MEMORY,
        "keep_warm_seconds": KEEP_WARM_SECONDS,
        "authorized": False,
        "volumes": [] if upstream_url() else [MODEL_VOLUME],
        "on_start": warmup_runtime,
        "env": {
            "OWNED_TTS_VARIANT": variant,
            "OWNED_TTS_GPU": gpu or "cpu",
            "OWNED_TTS_MODEL_DIR": "/models/dots/soar",
            "OWNED_TTS_KEEP_WARM_SECONDS": str(KEEP_WARM_SECONDS),
            **({"OWNED_TTS_UPSTREAM_URL": upstream_url()} if upstream_url() else {}),
        },
    }
    if gpu:
        spec["gpu"] = gpu
    if not upstream_url():
        spec["image"] = WORKER_IMAGE
    return spec


PREMIUM_SPEC = _asgi_spec("owned-dots-tts-premium", variant="soar", gpu_name=PREMIUM_GPU)
INTERACTIVE_SPEC = _asgi_spec("owned-dots-tts-interactive", variant="mf", gpu_name=INTERACTIVE_GPU)


@asgi(**PREMIUM_SPEC)
def premium(context=None):
    return _asgi_app(context)


@asgi(**INTERACTIVE_SPEC)
def interactive(context=None):
    return _asgi_app(context)
