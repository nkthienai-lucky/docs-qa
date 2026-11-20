"""Explicit serving and storage interlocks; importing this module starts nothing."""

from __future__ import annotations

import ipaddress
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .api import build_runtime_app
from .runtime import RuntimeConfigurationError, validate_runtime_profile
from .settings import RuntimeProfile, Settings, get_settings


def resolve_bind(settings: Settings, environ: Mapping[str, str]) -> tuple[str, int]:
    cloud = settings.runtime_profile == RuntimeProfile.CLOUD
    if cloud and not settings.runtime_deployment_approved:
        raise RuntimeConfigurationError("public deployment approval required")
    port = environ.get("PORT", "" if cloud else "8000")
    if (
        not isinstance(port, str)
        or not port.isascii()
        or not port.isdecimal()
        or not 1 <= int(port) <= 65535
    ):
        raise RuntimeConfigurationError("invalid assigned PORT")
    host = "0.0.0.0" if cloud else environ.get("HOST", "127.0.0.1")
    if not cloud:
        valid = host == "localhost"
        try:
            valid = valid or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if not valid:
            raise RuntimeConfigurationError("local serving requires loopback")
    return host, int(port)


def validate_upload_dir(settings: Settings) -> Path:
    path = Path(settings.runtime_upload_dir)
    cloud = settings.runtime_profile == RuntimeProfile.CLOUD
    valid = False
    try:
        if cloud:
            if (
                not path.is_absolute()
                or not path.is_dir()
                or any(p.is_symlink() for p in (path, *path.parents))
            ):
                raise ValueError
        else:
            path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path, prefix=".docsqa-write-check-", delete=True
        ) as sentinel:
            sentinel.write(b"check")
            sentinel.flush()
        valid = True
    except (OSError, ValueError):
        pass
    if not valid:
        raise RuntimeConfigurationError("upload directory validation failed")
    return path


def create_runtime_app():
    settings = get_settings()
    resolve_bind(settings, os.environ)
    validate_runtime_profile(settings)
    validate_upload_dir(settings)
    return build_runtime_app(settings)


def main() -> None:
    import uvicorn

    # Construct once; a failed interlock must never leave a listening socket.
    settings = get_settings()
    host, port = resolve_bind(settings, os.environ)
    app = create_runtime_app()
    uvicorn.run(app, host=host, port=port)
